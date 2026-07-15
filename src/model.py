import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import math


class PINN_TwoBranch_3D_Tscaled(nn.Module):
    """
    Decoupled parametric PINN for 3D transient heat conduction in metal AM.

    Architecture:
      - xt branch  : (x,y,z,t) -> e_xt  [2 hidden layers, width 30, embed dim 30]
      - lam branch : (rho,Cp,k) -> e_lam [2 hidden layers, width 30, embed dim 30]
      - fusion net : concat(e_xt, e_lam) -> T_raw [2 hidden layers, width 50]

    Output scaling:
      T = T_ref + T_max * ΔT_Ros(λ) * (Softplus(T_raw) + 1e-3)
    where ΔT_Ros(λ) = η·P / (2π·k·r_beam) is Rosenthal's analytical scale.
    """

    def __init__(
        self,
        xt_embed_dim=30,
        xt_hidden_layers=2,
        xt_width=30,
        lam_hidden_layers=2,
        lam_width=30,
        lam_embed_dim=30,
        fusion_hidden_layers=2,
        fusion_width=50,
        act=nn.Tanh,
        in_tf=None,
        out_tf=None,
        T_max=1.5,        # dimensionless scale factor on Rosenthal temperature
        T_ref=300.0,
        lambda_min=None,
        lambda_max=None,
        P_laser=500.0,    # laser power [W]
        v_scan=10.0,      # scanning speed [mm/s]
        r_beam=1.5,       # beam radius [mm]
        eta_abs=0.4,      # absorptivity [-]
    ):
        super().__init__()

        self.in_tf   = in_tf
        self.out_tf  = out_tf
        self.T_max   = T_max
        self.T_ref   = T_ref
        self.P_laser = P_laser
        self.v_scan  = v_scan
        self.r_beam  = r_beam
        self.eta_abs = eta_abs

        self.register_buffer("lam_min", lambda_min)
        self.register_buffer("lam_max", lambda_max)

        # Spatiotemporal branch: (x,y,z,t) -> e_xt
        xt_sizes = [4] + [xt_width] * xt_hidden_layers + [xt_embed_dim]
        self.xt_mlp = nn.Sequential()
        for i in range(len(xt_sizes) - 1):
            self.xt_mlp.add_module(f"xt_linear_{i}", nn.Linear(xt_sizes[i], xt_sizes[i + 1]))
            if i < len(xt_sizes) - 2:
                self.xt_mlp.add_module(f"xt_act_{i}", act())

        # Material branch: (rho,Cp,k) -> e_lam
        lam_sizes = [3] + [lam_width] * lam_hidden_layers + [lam_embed_dim]
        self.lam_feature_net = nn.Sequential()
        for i in range(len(lam_sizes) - 1):
            self.lam_feature_net.add_module(f"lamf_linear_{i}", nn.Linear(lam_sizes[i], lam_sizes[i + 1]))
            if i < len(lam_sizes) - 2:
                self.lam_feature_net.add_module(f"lamf_act_{i}", act())

        # Fusion network: concat(e_xt, e_lam) -> T_raw
        fuse_sizes = [xt_embed_dim + lam_embed_dim] + [fusion_width] * fusion_hidden_layers + [1]
        self.fusion_net = nn.Sequential()
        for i in range(len(fuse_sizes) - 1):
            self.fusion_net.add_module(f"fuse_linear_{i}", nn.Linear(fuse_sizes[i], fuse_sizes[i + 1]))
            if i < len(fuse_sizes) - 2:
                self.fusion_net.add_module(f"fuse_act_{i}", act())

    def rosenthal_deltaT(self, lam_raw):
        """
        Physics-guided temperature scale from Rosenthal's analytical solution.
        lam_raw: (N,3) = [rho, Cp, k] in internal units
        Returns ΔT = η·P / (2π·k·r_beam)  [K]
        """
        k = lam_raw[:, 2:3]
        return self.eta_abs * self.P_laser / (2.0 * math.pi * k * self.r_beam)

    def forward(self, X_input, return_raw=False):
        # Normalize (x,y,z,t) to [-1, 1]
        if self.in_tf:
            X_tf = X_input.clone()
            X_tf[:, :4] = self.in_tf(X_tf[:, :4])
            X_input = X_tf

        xt      = X_input[:, :4]   # (N,4)
        lam_raw = X_input[:, 4:7]  # (N,3) [rho, Cp, k] in internal units

        # Normalize lambda to [-1, 1] for network input
        lam_norm = 2.0 * (lam_raw - self.lam_min) / (self.lam_max - self.lam_min) - 1.0

        # Branch embeddings
        e_xt  = self.xt_mlp(xt)
        e_lam = self.lam_feature_net(lam_norm)

        # Fusion
        T_raw = self.fusion_net(torch.cat([e_xt, e_lam], dim=1))

        # Positive amplitude: Softplus + offset prevents collapse to zero
        A = F.softplus(T_raw) + 1e-3

        # Rosenthal-based material-dependent temperature scale
        T_max_lambda = self.rosenthal_deltaT(lam_raw)

        T = self.T_ref + self.T_max * T_max_lambda * A

        if return_raw:
            return T, T_raw, A, e_xt, e_lam, T_max_lambda
        return T

    def PDE(self, X_input):
        X_input.requires_grad_(True)
        T = self.forward(X_input)

        rho = X_input[:, 4:5]
        Cp  = X_input[:, 5:6]
        k   = X_input[:, 6:7]

        grad_T = grad(T, X_input, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_x = grad_T[:, 0:1]
        T_y = grad_T[:, 1:2]
        T_z = grad_T[:, 2:3]
        T_t = grad_T[:, 3:4]

        T_xx = grad(T_x, X_input, grad_outputs=torch.ones_like(T_x), create_graph=True)[0][:, 0:1]
        T_yy = grad(T_y, X_input, grad_outputs=torch.ones_like(T_y), create_graph=True)[0][:, 1:2]
        T_zz = grad(T_z, X_input, grad_outputs=torch.ones_like(T_z), create_graph=True)[0][:, 2:3]

        residual = rho * Cp * T_t - k * (T_xx + T_yy + T_zz)
        loss_res = torch.mean(residual ** 2)
        return loss_res, residual.detach().cpu().numpy()


def pde_loss_only(model, X_input):
    """PDE residual loss without numpy conversion (used during training)."""
    X = X_input.detach().clone().requires_grad_(True)
    T = model(X)

    rho = X[:, 4:5]
    Cp  = X[:, 5:6]
    k   = X[:, 6:7]

    grad_T = grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grad_T[:, 0:1]
    T_y = grad_T[:, 1:2]
    T_z = grad_T[:, 2:3]
    T_t = grad_T[:, 3:4]

    T_xx = grad(T_x, X, grad_outputs=torch.ones_like(T_x), create_graph=True)[0][:, 0:1]
    T_yy = grad(T_y, X, grad_outputs=torch.ones_like(T_y), create_graph=True)[0][:, 1:2]
    T_zz = grad(T_z, X, grad_outputs=torch.ones_like(T_z), create_graph=True)[0][:, 2:3]

    residual = rho * Cp * T_t - k * (T_xx + T_yy + T_zz)
    return torch.mean(residual ** 2)
