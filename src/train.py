import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import random
import logging
import os
import re
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from utils import sampling_uniform
from model import PINN_TwoBranch_3D_Tscaled, pde_loss_only


# ================================================================
# Utility: parse material properties (rho, Cp, k) from directory name
# ================================================================
def parse_lambda_from_path(path: str):
    """
    Parse (rho_si, cp_si, k_si) from a path containing a tuple like (8960,385,401).
    Input units: rho [kg/m^3], Cp [J/(kg·K)], k [W/(m·K)]
    Output units: rho [g/mm^3], Cp [J/(g·K)], k [W/(mm·K)]
    """
    m = re.search(r"\(([^)]+)\)", path)
    if m is None:
        raise ValueError(f"Cannot find (rho,Cp,k) tuple in path: {path}")
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Tuple must have 3 values, got {parts} from: {path}")
    rho_si, cp_si, k_si = map(float, parts)
    rho = rho_si / 1e6
    cp  = cp_si  / 1000.0
    k   = k_si   / 1000.0
    return rho, cp, k


def make_input_transform(X_min, X_max):
    """Normalize [x, y, z, t] from physical domain to [-1, 1]."""
    def _tf(X):
        return 2.0 * (X - X_min) / (X_max - X_min) - 1.0
    return _tf


# ================================================================
# Collocation point generation (manual multi-resolution grid)
# ================================================================
def generate_points_manual(cfg, x_min, x_max, device, p=None, f=None):
    """
    Generate collocation points for domain PDE residual, boundary
    conditions on all six faces, and the initial condition at t=0.
    Spatial coordinates in [mm], time in [s].
    """
    if p is None:
        p = []
    if f is None:
        f = []

    colloc = cfg.collocation
    laser  = cfg.laser
    bc     = cfg.bc

    # Time grid [s]
    t = np.linspace(x_min[3] + colloc.t_start_offset, x_max[3], colloc.t_samples)

    # Six boundary faces
    bound_x_neg, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '-x', t)
    bound_x_pos, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '+x', t)
    bound_y_neg, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '-y', t)
    bound_y_pos, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '+y', t)
    bound_z_neg, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '-z', t)
    bound_z_pos, _ = sampling_uniform(colloc.bc_density, x_min, x_max, '+z', t)

    # Extra top-surface (z=z_max) points near the moving laser spot
    bound_z_pos_more = []
    for ti in t:
        if ti <= laser.t_end:
            zi, _ = sampling_uniform(
                colloc.bc_top_density,
                [max(laser.x0 + ti * laser.v_scan - colloc.bc_top_radius_mult * laser.r_beam, x_min[0]),
                 max(x_min[1], laser.y0 - colloc.bc_top_radius_mult * laser.r_beam),
                 x_min[2]],
                [min(laser.x0 + ti * laser.v_scan + colloc.bc_top_radius_mult * laser.r_beam, x_max[0]),
                 min(x_max[1], laser.y0 + colloc.bc_top_radius_mult * laser.r_beam),
                 x_max[2]],
                '+z', [ti]
            )
            bound_z_pos_more.append(zi)
    bound_z_pos = np.vstack((bound_z_pos, np.vstack(bound_z_pos_more)))

    # Domain interior: three-region z-stratified sampling
    domain_pts1, _ = sampling_uniform(
        colloc.domain_density_low,
        [x_min[0], x_min[1], x_min[2]],
        [x_max[0], x_max[1], x_max[2] - colloc.z_mid_thickness],
        'domain', t
    )
    domain_pts2, _ = sampling_uniform(
        colloc.domain_density_mid,
        [x_min[0], x_min[1], x_max[2] - colloc.z_mid_thickness + colloc.z_mid_overlap],
        [x_max[0], x_max[1], x_max[2] - colloc.z_top_thickness],
        'domain', t
    )
    domain_pts3 = []
    for ti in t:
        di, _ = sampling_uniform(
            colloc.domain_density_top,
            [x_min[0], x_min[1], x_max[2] - colloc.z_top_thickness + colloc.z_top_overlap],
            [x_max[0], x_max[1], x_max[2]],
            'domain', [ti]
        )
        domain_pts3.append(di)
    domain_pts = np.vstack((domain_pts1, domain_pts2, np.vstack(domain_pts3)))

    # Initial condition points (t = 0)
    init_pts1, _ = sampling_uniform(
        colloc.init_density,
        [x_min[0], x_min[1], x_min[2]],
        [x_max[0], x_max[1], x_max[2]],
        'domain', [0], e=0
    )
    init_pts2, _ = sampling_uniform(
        colloc.init_focus_density,
        [laser.x0 - colloc.init_focus_xy_radius, laser.y0 - colloc.init_focus_xy_radius,
         x_max[2] - colloc.init_focus_z_thickness],
        [laser.x0 + colloc.init_focus_xy_radius, laser.y0 + colloc.init_focus_xy_radius, x_max[2]],
        'domain', [0]
    )
    init_pts = np.vstack((init_pts1, init_pts2))

    # Convert to tensors
    p.extend([
        torch.tensor(bound_x_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_x_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(init_pts,    requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(domain_pts,  requires_grad=True, dtype=torch.float).to(device),
    ])
    f.extend([
        ['BC', '-x'], ['BC', '+x'],
        ['BC', '-y'], ['BC', '+y'],
        ['BC', '-z'], ['BC', '+z'],
        ['IC', bc.t_ref],
        ['domain'],
    ])
    return p, f


# ================================================================
# Loss helpers
# ================================================================
def relative_l2_error(pred, target, eps=1e-12):
    num = torch.sqrt(torch.sum((pred - target) ** 2))
    den = torch.sqrt(torch.sum(target ** 2)) + eps
    return num / den


def loss(f, target=None):
    if target is None:
        return torch.sum(torch.square(f)) / f.shape[0]
    if isinstance(target, float):
        return torch.sum(torch.square(f - target)) / f.shape[0]
    return nn.MSELoss()(f, target)


def safe_mse(x):
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=x.device)
    return torch.mean(x ** 2)


def safe_rel_l2(pred, target, eps=1e-12):
    if pred.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    num = torch.sqrt(torch.sum((pred - target) ** 2))
    den = torch.sqrt(torch.sum(target ** 2)) + eps
    return num / den


# ================================================================
# Boundary condition residuals
# ================================================================
def radiation_flux(T, T_ref, Rboltz, emiss):
    T_rad = torch.clamp(T, min=0.0, max=100000.0)
    return Rboltz * emiss * (T_rad ** 4 - T_ref ** 4)


def BC_residual(model, X_input, loc, cfg):
    bc    = cfg.bc
    laser = cfg.laser
    X = X_input.clone().detach().requires_grad_(True)
    T = model(X)

    x = X[:, 0:1]; y = X[:, 1:2]; z = X[:, 2:3]; t = X[:, 3:4]
    k = X[:, 6:7]

    grad_T = grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grad_T[:, 0:1]; T_y = grad_T[:, 1:2]; T_z = grad_T[:, 2:3]; T_t = grad_T[:, 3:4]

    rad = radiation_flux(T, bc.t_ref, bc.rboltz, bc.emiss)

    if loc == '-x':
        return k * T_x - bc.h_conv * (T - bc.t_ref) - rad
    if loc == '+x':
        return -k * T_x - bc.h_conv * (T - bc.t_ref) - rad
    if loc == '-y':
        return k * T_y - bc.h_conv * (T - bc.t_ref) - rad
    if loc == '+y':
        return -k * T_y - bc.h_conv * (T - bc.t_ref) - rad
    if loc == '-z':
        return T_t
    if loc == '+z':
        q = (2 * laser.p_laser * laser.eta_abs / torch.pi / (laser.r_beam ** 2)
             * torch.exp(-2 * ((x - laser.x0 - laser.v_scan * t) ** 2 + (y - laser.y0) ** 2)
                         / (laser.r_beam ** 2))
             * (t <= laser.t_end) * (t > 0))
        return -k * T_z - bc.h_conv * (T - bc.t_ref) - rad + q
    raise ValueError(f"Unknown BC location: {loc}")


# ================================================================
# Lambda (material property) sampling
# ================================================================
def sample_uniform_lambda(lambda_bounds, size):
    out = []
    for (lam_min, lam_max) in lambda_bounds:
        out.append(np.random.uniform(lam_min, lam_max, (size, 1)))
    return np.hstack(out).astype(np.float32)


# ================================================================
# Trainer
# ================================================================
class Trainer:
    def __init__(self, model, xt_bounds, lambda_bounds, device="cpu", cfg=None, x_min=None, x_max=None):
        self.model = model
        self.xt_bounds = xt_bounds
        self.lambda_bounds = lambda_bounds
        self.device = device
        self.cfg = cfg
        self.x_min = x_min
        self.x_max = x_max

    def _sample_lambda_uniform(self, size):
        return sample_uniform_lambda(self.lambda_bounds, size)

    def _build_point_sets(self, base_points_4d, flags):
        """Attach uniformly sampled lambda to each set of 4D collocation points."""
        point_sets = []
        n_bc = n_ic = n_pde = 0

        for pts4, flag in zip(base_points_4d, flags):
            n_i   = int(pts4.shape[0])
            lam_np = self._sample_lambda_uniform(n_i)
            lam_t  = torch.tensor(lam_np, dtype=torch.float32, device=self.device)
            x7 = torch.cat([pts4.to(self.device), lam_t], dim=1).requires_grad_(True)
            point_sets.append(x7)

            if flag[0] == "BC":
                n_bc += n_i
            elif flag[0] == "IC":
                n_ic += n_i
            elif flag[0] == "domain":
                n_pde += n_i

        return point_sets, n_bc, n_ic, n_pde

    def run_training_loop(
        self,
        E_final=1000,
        test_in=None,
        test_out=None,
        info_num=100,
        adam_epochs=None,
        bc_only_epochs=None,
        run_dir=None,
        ic_weight=1e-4,
        adam_lr=2e-4,
        adam_bc_sub=12000,
        adam_ic_sub=6000,
        adam_pde_sub=20000,
        lbfgs_bc_sub=8000,
        lbfgs_ic_sub=4000,
        lbfgs_pde_sub=12000,
        lbfgs_refresh_every=200,
        lbfgs_lr=1.0,
        lbfgs_max_iter=50,
        lbfgs_history_size=50,
        lbfgs_line_search="strong_wolfe",
    ):
        cfg = self.cfg
        if cfg is None:
            raise ValueError("cfg is required for run_training_loop")

        bc = cfg.bc

        # Move validation data to CPU to save GPU memory during training
        if (test_in is not None) and (test_out is not None):
            test_in  = [x.detach().cpu() for x in test_in]
            test_out = [y.detach().cpu() for y in test_out]

        print(f"[Training] {E_final} total epochs")

        # Reset model weights before training
        def reset_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.model.apply(reset_weights)

        # Generate 4D collocation points (x, y, z, t) and attach lambda
        base_points_4d, flags = generate_points_manual(cfg, self.x_min, self.x_max, self.device)
        point_sets, n_bc, n_ic, n_pde = self._build_point_sets(base_points_4d, flags)
        print(f"[Collocation] n_BC={n_bc}, n_IC={n_ic}, n_PDE={n_pde}")

        l_history   = []
        err_history = []

        def run_validation(ep):
            if (test_in is None) or (test_out is None):
                return None, None
            if not (((ep + 1) % info_num == 0) or (ep == 0)):
                return None, None

            self.model.eval()
            with torch.no_grad():
                n_val   = len(test_in)
                mse_arr = np.full((n_val,), np.nan, dtype=np.float64)
                rel_arr = np.full((n_val,), np.nan, dtype=np.float64)

                for i, (x_cpu, t_cpu) in enumerate(zip(test_in, test_out)):
                    x_test = x_cpu.to(self.device, non_blocking=True)
                    t_test = t_cpu.to(self.device, non_blocking=True)
                    t_pred = self.model(x_test)
                    mse_arr[i] = loss(t_pred, t_test).item()
                    rel_arr[i] = relative_l2_error(t_pred, t_test).item()
                    print(f"  [Val {i}] Epoch {ep+1}/{E_final}  MSE={mse_arr[i]:.3e}  RelL2={rel_arr[i]:.3e}")
                    del x_test, t_test, t_pred

                test_mse = float(np.nanmean(mse_arr))
                test_rel = float(np.nanmean(rel_arr))
                row = [float(ep + 1), test_mse, test_rel]
                for i in range(n_val):
                    row += [float(mse_arr[i]), float(rel_arr[i])]
                err_history.append(row)

            self.model.train()
            return test_mse, test_rel

        # Epoch budget split between Adam and L-BFGS
        if (adam_epochs is None) or (adam_epochs < 0):
            adam_total = min(2000, E_final)
        else:
            adam_total = min(int(adam_epochs), E_final)

        if (bc_only_epochs is None) or (bc_only_epochs < 0):
            bc_only_total = min(200, adam_total)
        else:
            bc_only_total = min(int(bc_only_epochs), adam_total)

        lbfgs_total = E_final - adam_total

        adam_bc_sub   = int(adam_bc_sub)
        adam_ic_sub   = int(adam_ic_sub)
        adam_pde_sub  = int(adam_pde_sub)
        n_bc_sub      = int(lbfgs_bc_sub)
        n_ic_sub      = int(lbfgs_ic_sub)
        n_pde_sub     = int(lbfgs_pde_sub)
        refresh_every = int(lbfgs_refresh_every)

        def subsample(x, m):
            n = x.shape[0]
            if m >= n:
                return x
            idx = torch.randperm(n, device=x.device)[:m]
            return x[idx]

        # ---- Adam phase ----
        if adam_total > 0:
            adam = torch.optim.Adam(self.model.parameters(), lr=adam_lr)
            for ep in range(adam_total):
                adam.zero_grad(set_to_none=True)

                l_bc  = torch.tensor(0.0, device=self.device)
                l_ic  = torch.tensor(0.0, device=self.device)
                l_pde = torch.tensor(0.0, device=self.device)

                for x7, flag in zip(point_sets, flags):
                    if flag[0] == "BC":
                        x    = subsample(x7, adam_bc_sub)
                        r_bc = BC_residual(self.model, x, flag[1], cfg)
                        l_bc += torch.mean(r_bc ** 2) * x.shape[0] / max(n_bc, 1)
                    elif flag[0] == "IC":
                        x    = subsample(x7, adam_ic_sub)
                        t_ic = self.model(x)
                        l_ic += torch.mean((t_ic - bc.t_ref) ** 2) * x.shape[0] / max(n_ic, 1)
                    elif flag[0] == "domain":
                        x     = subsample(x7, adam_pde_sub)
                        l_pde += pde_loss_only(self.model, x) * x.shape[0] / max(n_pde, 1)

                # Curriculum: BC-only for the first bc_only_total epochs
                cost = l_bc if ep < bc_only_total else l_bc + ic_weight * l_ic + l_pde
                cost.backward()
                adam.step()

                l_history.append([cost.item(), l_bc.item(), l_ic.item(), l_pde.item()])

                if (ep + 1) % 100 == 0:
                    print(f"[Epoch {ep+1}] (Adam)  BC={l_bc:.3e}  IC={l_ic:.3e}  PDE={l_pde:.3e}  Total={cost:.3e}")

                run_validation(ep)

        # ---- L-BFGS phase ----
        if lbfgs_total > 0:

            def build_lbfgs_sets():
                sets = []
                for x7, flag in zip(point_sets, flags):
                    n = x7.shape[0]
                    if flag[0] == "BC":
                        m = min(n, n_bc_sub)
                    elif flag[0] == "IC":
                        m = min(n, n_ic_sub)
                    elif flag[0] == "domain":
                        m = min(n, n_pde_sub)
                    else:
                        continue
                    idx = torch.randperm(n, device=self.device)[:m]
                    sets.append((x7[idx].detach().clone(), flag))
                n_bc_lb  = sum(x.shape[0] for x, f in sets if f[0] == "BC")
                n_ic_lb  = sum(x.shape[0] for x, f in sets if f[0] == "IC")
                n_pde_lb = sum(x.shape[0] for x, f in sets if f[0] == "domain")
                return sets, n_bc_lb, n_ic_lb, n_pde_lb

            def make_lbfgs():
                ls = lbfgs_line_search
                if isinstance(ls, str) and ls.strip().lower() in ["", "none", "null"]:
                    ls = None
                return torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=lbfgs_lr,
                    max_iter=int(lbfgs_max_iter),
                    history_size=int(lbfgs_history_size),
                    line_search_fn=ls,
                )

            lbfgs_sets, n_bc_lb, n_ic_lb, n_pde_lb = build_lbfgs_sets()
            lbfgs = make_lbfgs()
            last  = {}

            def closure():
                lbfgs.zero_grad(set_to_none=True)

                l_bc  = torch.tensor(0.0, device=self.device)
                l_ic  = torch.tensor(0.0, device=self.device)
                l_pde = torch.tensor(0.0, device=self.device)

                for x_sub, flag in lbfgs_sets:
                    x = x_sub.detach().clone()
                    if flag[0] == "BC":
                        r_bc  = BC_residual(self.model, x, flag[1], cfg)
                        l_bc += torch.mean(r_bc ** 2) * x.shape[0] / max(n_bc_lb, 1)
                    elif flag[0] == "IC":
                        t_ic  = self.model(x)
                        l_ic += torch.mean((t_ic - bc.t_ref) ** 2) * x.shape[0] / max(n_ic_lb, 1)
                    elif flag[0] == "domain":
                        l_pde += pde_loss_only(self.model, x) * x.shape[0] / max(n_pde_lb, 1)

                cost = l_bc + ic_weight * l_ic + l_pde
                cost.backward()
                last["cost"]  = float(cost.item())
                last["l_BC"]  = float(l_bc.item())
                last["l_IC"]  = float(l_ic.item())
                last["l_PDE"] = float(l_pde.item())
                return cost

            for k in range(lbfgs_total):
                ep = adam_total + k

                # Refresh mini-batch subset and optimizer periodically
                if (k % refresh_every) == 0 and k > 0:
                    lbfgs_sets, n_bc_lb, n_ic_lb, n_pde_lb = build_lbfgs_sets()
                    lbfgs = make_lbfgs()
                    last.clear()
                    print(f"[L-BFGS] refreshed subsets at epoch {ep+1}")

                lbfgs.step(closure)
                l_history.append([last["cost"], last["l_BC"], last["l_IC"], last["l_PDE"]])

                if (ep + 1) % 100 == 0:
                    print(f"[Epoch {ep+1}] (L-BFGS)  BC={last['l_BC']:.3e}  IC={last['l_IC']:.3e}  "
                          f"PDE={last['l_PDE']:.3e}  Total={last['cost']:.3e}")

                run_validation(ep)

        return l_history, err_history


# ================================================================
# run_training: model setup, training, and saving results
# ================================================================
def run_training(args, xt_bounds, lambda_bounds, device, x_min, x_max, X_min, X_max):
    base     = args.base
    training = args.training
    schedule = args.schedule
    bc       = args.bc
    laser    = args.laser
    seed     = base.seed
    output   = base.output

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed: {seed}")

    run_dir = os.path.join("results", "bareplate", "proposed", output, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    lam_min = torch.tensor([lambda_bounds[0][0], lambda_bounds[1][0], lambda_bounds[2][0]], device=device)
    lam_max = torch.tensor([lambda_bounds[0][1], lambda_bounds[1][1], lambda_bounds[2][1]], device=device)

    in_tf = make_input_transform(X_min, X_max)

    model = PINN_TwoBranch_3D_Tscaled(
        in_tf=in_tf,
        out_tf=None,
        T_max=1.5,
        T_ref=bc.t_ref,
        lambda_min=lam_min,
        lambda_max=lam_max,
        P_laser=laser.p_laser,
        v_scan=laser.v_scan,
        r_beam=laser.r_beam,
        eta_abs=laser.eta_abs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params}")
    print("\n==================== Model Architecture ====================")
    print(model)
    print("=============================================================\n")

    trainer = Trainer(
        model=model,
        xt_bounds=xt_bounds,
        lambda_bounds=lambda_bounds,
        device=device,
        cfg=args,
        x_min=x_min,
        x_max=x_max,
    )

    # Load validation data
    X_tests     = []
    T_tests     = []
    valid_names = []

    for vp in args.validation.valid:
        data_valid = np.load(vp)
        xt_np = data_valid[:, 0:4].astype(np.float32)
        T_np  = data_valid[:, 4:5].astype(np.float32)
        N = xt_np.shape[0]

        rho_val, Cp_val, k_val = parse_lambda_from_path(vp)
        rho_np = np.full((N, 1), rho_val, dtype=np.float32)
        Cp_np  = np.full((N, 1), Cp_val,  dtype=np.float32)
        k_np   = np.full((N, 1), k_val,   dtype=np.float32)
        X_np   = np.concatenate([xt_np, rho_np, Cp_np, k_np], axis=1)

        X_tests.append(torch.tensor(X_np, dtype=torch.float32, device=device))
        T_tests.append(torch.tensor(T_np, dtype=torch.float32, device=device))
        valid_names.append(os.path.basename(vp))
        print(f"[Val] {os.path.basename(vp)} -> rho={rho_val:.4e}, Cp={Cp_val:.4e}, k={k_val:.4e}")

    # Train
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    l_history, err_history = trainer.run_training_loop(
        test_in=X_tests,
        test_out=T_tests,
        E_final=base.epoch_final,
        info_num=training.info_num,
        adam_epochs=schedule.adam_epochs,
        bc_only_epochs=schedule.bc_only_epochs,
        ic_weight=schedule.ic_weight,
        adam_lr=schedule.adam_lr,
        run_dir=run_dir,
        adam_bc_sub=schedule.adam_bc_sub,
        adam_ic_sub=schedule.adam_ic_sub,
        adam_pde_sub=schedule.adam_pde_sub,
        lbfgs_bc_sub=schedule.lbfgs_bc_sub,
        lbfgs_ic_sub=schedule.lbfgs_ic_sub,
        lbfgs_pde_sub=schedule.lbfgs_pde_sub,
        lbfgs_refresh_every=schedule.lbfgs_refresh_every,
        lbfgs_lr=schedule.lbfgs_lr,
        lbfgs_max_iter=schedule.lbfgs_max_iter,
        lbfgs_history_size=schedule.lbfgs_history_size,
        lbfgs_line_search=schedule.lbfgs_line_search,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Total training time: {time.time() - t_start:.2f} s")

    # Save outputs
    model_path    = os.path.join(run_dir, "model.pt")
    loss_npy_path = os.path.join(run_dir, "loss_history.npy")
    err_npy_path  = os.path.join(run_dir, "err_history.npy")
    loss_fig_path = os.path.join(run_dir, "loss_history.png")

    torch.save(model.state_dict(), model_path)
    np.save(loss_npy_path, l_history)
    np.save(err_npy_path,  err_history)

    with open(os.path.join(run_dir, "valid_names.txt"), "w") as fh:
        for i, nm in enumerate(valid_names):
            fh.write(f"{i}\t{nm}\n")

    # Loss curve
    l_arr = np.array(l_history)
    plt.figure()
    plt.semilogy(l_arr[:, 1], label="BC loss")
    plt.semilogy(l_arr[:, 2], label="IC loss")
    plt.semilogy(l_arr[:, 3], label="PDE loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("PINN training loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()

    print(f"[INFO] Model saved:       {model_path}")
    print(f"[INFO] Loss history:      {loss_npy_path}")
    print(f"[INFO] Error history:     {err_npy_path}")
    print(f"[INFO] Loss plot:         {loss_fig_path}")


# ================================================================
# Entry point
# ================================================================
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Parametric PINN training")

    base     = cfg.base
    domain   = cfg.domain
    material = cfg.material
    device   = torch.device(f"cuda:{base.device}")

    logging.info(f"Device: {device}")

    # Spatiotemporal domain [mm, s]
    xb = domain.xt_bounds
    XT_BOUNDS = [
        (xb[0], xb[1]),   # x
        (xb[2], xb[3]),   # y
        (xb[4], xb[5]),   # z
        (xb[6], xb[7]),   # t
    ]

    x_min = np.array([XT_BOUNDS[0][0], XT_BOUNDS[1][0], XT_BOUNDS[2][0], XT_BOUNDS[3][0]])
    x_max = np.array([XT_BOUNDS[0][1], XT_BOUNDS[1][1], XT_BOUNDS[2][1], XT_BOUNDS[3][1]])
    X_min = torch.tensor(x_min, dtype=torch.float).to(device)
    X_max = torch.tensor(x_max, dtype=torch.float).to(device)

    # Material property bounds in SI, converted to internal units
    RHO_BOUNDS_SI = (material.rho_bounds_si[0], material.rho_bounds_si[1])   # kg/m^3
    CP_BOUNDS_SI  = (material.cp_bounds_si[0],  material.cp_bounds_si[1])    # J/(kg·K)
    K_BOUNDS_SI   = (material.k_bounds_si[0],   material.k_bounds_si[1])     # W/(m·K)

    # Internal units: rho [g/mm^3], Cp [J/(g·K)], k [W/(mm·K)]
    LAMBDA_BOUNDS = [
        (RHO_BOUNDS_SI[0] / 1e6,    RHO_BOUNDS_SI[1] / 1e6),
        (CP_BOUNDS_SI[0]  / 1000.0, CP_BOUNDS_SI[1]  / 1000.0),
        (K_BOUNDS_SI[0]   / 1000.0, K_BOUNDS_SI[1]   / 1000.0),
    ]

    run_training(cfg, XT_BOUNDS, LAMBDA_BOUNDS, device, x_min, x_max, X_min, X_max)


if __name__ == "__main__":
    main()
