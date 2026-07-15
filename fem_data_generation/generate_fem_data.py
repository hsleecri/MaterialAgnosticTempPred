"""Generate FEM ground-truth temperature data for the bare-plate numerical benchmark in metal AM.

A single-track laser scan on a bare plate is simulated with JAX-AM
(https://github.com/CMSL-HKUST/jax-am), and the resulting temperature
snapshots are exported as a NumPy array of shape (N, 5) with columns
[x(mm), y(mm), z(mm), t(s), T(K)] — the format expected by the PINN
training code in this repository.

Usage (from this directory, with jax-am installed):

    python generate_fem_data.py --material Copper
    python generate_fem_data.py --material SS316L
    python generate_fem_data.py --rho 4430 --cp 560 --k 6.7 --name MyAlloy

Select the GPU via the CUDA_VISIBLE_DEVICES environment variable.
"""
import argparse
import glob
import os

import numpy as onp
import jax.numpy as np
import meshio

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

from models_bareplate import Thermal, initialize_hash_map, get_active_mesh


# ---------------------------------------------------------------------
# Material presets: name -> (rho [kg/m^3], Cp [J/(kg*K)], k [W/(m*K)])
# ---------------------------------------------------------------------
MATERIALS = {
    "Ti_6AL_4V":  (4430.0, 560.0, 6.7),
    "Inconel718": (8220.0, 435.0, 11.4),
    "SS316L":     (8000.0, 500.0, 16.0),
    "AlSi10Mg":   (2670.0, 950.0, 150.0),
    "Copper":     (8960.0, 385.0, 401.0),
}

SAVE_VTU = True
SAVE_SNAPSHOT_EVERY = 10
VERBOSE = True


def convert_vtu_to_numpy(vtu_dir, output_path, dt):
    """Read u_*.vtu files in vtu_dir and stack them into a single array
    with columns [x(mm), y(mm), z(mm), t(s), T(K)].
    """
    vtu_files = sorted(glob.glob(os.path.join(vtu_dir, "u_*.vtu")))
    print(f"[VTU->NumPy] Found {len(vtu_files)} VTU files")

    all_data = []

    for file in vtu_files:
        mesh = meshio.read(file)

        candidate_fields = ["u", "sol", "T", "temperature", "temp", "field", "scalar"]
        found_field = None
        for cand in candidate_fields:
            if cand in mesh.point_data.keys():
                found_field = cand
                break

        if found_field is None:
            raise ValueError(
                f"No temperature field found in {file}. Available: {list(mesh.point_data.keys())}"
            )

        T = mesh.point_data[found_field].reshape(-1, 1)

        xyz_mm = mesh.points * 1000.0
        step = int(file.split("_")[-1].split(".")[0])
        t = step * dt
        t_col = onp.ones((xyz_mm.shape[0], 1)) * t

        data = onp.hstack([xyz_mm, t_col, T])
        all_data.append(data)

        print(f"[OK] {file}: field='{found_field}', t={t:.4f}s")

    all_data = onp.vstack(all_data)
    onp.save(output_path, all_data)
    print(f"[VTU->NumPy] Saved: {output_path}, shape={all_data.shape}")
    return all_data


def bare_plate_single_track(name, rho, Cp, k, out_root):
    # ------------------------------
    # Process parameters (fixed across all materials in the paper)
    # ------------------------------
    t_total = 3.0          # total simulation time [s]
    t_end_laser = 3.0      # laser switch-off time [s]
    vel = 0.01             # scanning speed [m/s]
    dt = 1e-2              # time step [s]

    T0 = 300.0             # initial / ambient temperature [K]
    h = 50.0               # convection coefficient [W/(m^2*K)]
    rb = 1.5e-3            # laser beam radius [m]
    eta = 0.4              # laser absorptivity
    P = 500.0              # laser power [W]

    SIGMA_SB = 5.6704e-8   # Stefan-Boltzmann constant [W/(m^2*K^4)]
    EMISS = 0.3            # surface emissivity
    T_AMB = T0

    vec = 1
    dim = 3
    ele_type = 'HEX8'

    ts = np.arange(0.0, t_total + 1e-12, dt)
    problem_name = "bare_plate"

    # ------------------------------
    # Output directories
    # ------------------------------
    run_dir = os.path.join(out_root, f"{name}_({rho:g},{Cp:g},{k:g})")
    data_dir = os.path.join(run_dir, 'data')
    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)

    # Remove VTU files from a previous run
    for f in glob.glob(os.path.join(vtk_dir, f"{problem_name}/*")):
        try:
            os.remove(f)
        except OSError:
            pass
    os.makedirs(os.path.join(vtk_dir, problem_name), exist_ok=True)

    # -------------------------
    # Mesh
    # -------------------------
    Nx, Ny, Nz = 150, 30, 18
    Lx, Ly, Lz = 40e-3, 10e-3, 6e-3

    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    full_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict["hexahedron"])

    active_cell_truth_tab = onp.ones(len(full_mesh.cells), dtype=bool)
    active_mesh, pts_map, cells_map = get_active_mesh(full_mesh, active_cell_truth_tab)

    external_faces, cells_face, hash_map, inner_faces, all_faces = initialize_hash_map(
        full_mesh, active_cell_truth_tab, cells_map, ele_type
    )

    sol = T0 * np.ones((len(active_mesh.points), vec))

    # -------------------------
    # Global time variable (read by the Neumann BC closures below;
    # updated each step of the time-marching loop)
    # -------------------------
    global t_now_sim
    t_now_sim = 0.0

    # -------------------------
    # Neumann BCs
    # -------------------------
    def neumann_top(point, old_T):
        global t_now_sim

        if t_now_sim > t_end_laser:
            q_laser = 0.0
        else:
            d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
            q_laser = 2 * eta * P / (np.pi * rb**2) * np.exp(-2 * d2 / rb**2)

        q_conv_rad = h * (T_AMB - old_T[0]) \
                     + SIGMA_SB * EMISS * (T_AMB**4 - old_T[0]**4)

        return np.array([q_laser + q_conv_rad])

    def neumann_walls(point, old_T):
        q_conv_rad = h * (T_AMB - old_T[0]) \
                     + SIGMA_SB * EMISS * (T_AMB**4 - old_T[0]**4)
        return np.array([q_conv_rad])

    neumann_bc_info = [None, [neumann_top, neumann_walls]]

    # -------------------------
    # Thermal problem setup
    # -------------------------
    problem = Thermal(
        active_mesh,
        vec=vec,
        dim=dim,
        neumann_bc_info=neumann_bc_info,
        additional_info=(sol, rho, Cp, dt, external_faces, k)
    )

    if SAVE_VTU:
        save_sol(problem, sol, os.path.join(vtk_dir, f"{problem_name}/u_00000.vtu"))

    # -------------------------
    # Laser start position
    # -------------------------
    x0_m = 5e-3
    y0_m = Ly / 2.0
    z_top = Lz

    # -------------------------
    # Time-marching loop
    # -------------------------
    for i in range(1, len(ts)):
        t_now_sim = float(ts[i])
        x_now = float(x0_m + vel * t_now_sim)
        global laser_center
        laser_center = np.array([x_now, y0_m, z_top])

        if VERBOSE:
            print(f"[Step {i}] t={t_now_sim:.4f}s, laser_x={x_now*1000:.2f} mm")

        sol = solver(problem)
        problem.update_int_vars(sol)

        if SAVE_VTU and (i % SAVE_SNAPSHOT_EVERY == 0):
            save_sol(problem, sol, os.path.join(vtk_dir, f"{problem_name}/u_{i:05d}.vtu"))

        if x_now > Lx + 5e-3:
            print("[Stopping] Laser passed the simulated region.")
            break

    print("\n=== Simulation Finished ===")
    print("Starting VTU -> NumPy conversion...")

    output_npy = os.path.join(data_dir, "data.npy")
    convert_vtu_to_numpy(os.path.join(vtk_dir, problem_name), output_npy, dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", choices=sorted(MATERIALS.keys()),
                        help="material preset used in the paper")
    parser.add_argument("--rho", type=float, help="density [kg/m^3] (overrides preset)")
    parser.add_argument("--cp", type=float, help="specific heat capacity [J/(kg*K)] (overrides preset)")
    parser.add_argument("--k", type=float, help="thermal conductivity [W/(m*K)] (overrides preset)")
    parser.add_argument("--name", default=None,
                        help="output folder name (defaults to the material preset name)")
    parser.add_argument("--out-root", default="FEM_data",
                        help="root output directory (default: FEM_data)")
    args = parser.parse_args()

    if args.material is not None:
        rho, cp, k = MATERIALS[args.material]
        name = args.name or args.material
    elif None not in (args.rho, args.cp, args.k):
        rho, cp, k = args.rho, args.cp, args.k
        name = args.name or "Custom"
    else:
        parser.error("Provide --material, or all of --rho/--cp/--k.")

    rho = args.rho if args.rho is not None else rho
    cp = args.cp if args.cp is not None else cp
    k = args.k if args.k is not None else k

    print(f"Material: {name} (rho={rho}, Cp={cp}, k={k})")
    bare_plate_single_track(name, rho, cp, k, args.out_root)
