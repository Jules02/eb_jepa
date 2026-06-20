import torch
import torch.nn as nn
import torch.nn.functional as F


def sq_loss(x, y, reduction="mean"):
    """Simple square loss (MSE)."""
    return nn.functional.mse_loss(x, y, reduction=reduction)


def square_cost_seq(state, predi):
    """Square loss between two [B, C, T, H, W] sequences."""
    return sq_loss(state, predi)


class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return square_cost_seq(state, predi)


class VCLoss(nn.Module):
    """Variance-Covariance loss attracting means to zero and covariance to identity."""

    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)  # [B*T*H*W, C']

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted_loss = std_loss + cov_loss
        loss_dict = {
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }
        return loss, total_unweighted_loss, loss_dict


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss


class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t


class InverseDynamicsLoss(torch.nn.Module):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions.
        Args:
            idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1) and predicts action
        """
        super().__init__()
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: [T, B, D] - States across time steps
            actions: [B, A, T] - Ground truth actions between consecutive states
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        t, b, d = x.shape

        states_t = x[:-1].transpose(0, 1)  # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1)  # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d)  # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d)  # [B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(
            -1, actions.size(1)
        )  # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss


class VC_IDM_Sim_Regularizer(torch.nn.Module):
    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
        reg_type: str = "vicreg",
        sigreg_coeff: float = 0.0,
        sigreg_num_slices: int = 256,
        tdist_coeff: float = 0.0,
        dist_split_frac: float = 0.0,
        cross_cov_coeff: float = 0.0,
    ):
        """
        Composite Regularizer combining multiple losses

        This is a composite loss that combines:
        - Hinge Standard Deviation Loss
        - Covariance Decorrelation Loss
        - Temporal Similarity Loss
        - Inverse Dynamics Model Loss

        Args:
            cov_coeff (float): Weight for covariance loss
            std_coeff (float): Weight for std hinge loss
            sim_coeff_t (float): Weight for temporal similarity loss
            idm_coeff (float): Weight for inverse dynamics loss
            idm (nn.Module): Inverse dynamics model
            std_margin (float): Minimum desired std per feature
            first_t_only (bool): Use only first time slice for std/cov loss
            projector (nn.Module): Optional projection layer
            spatial_as_samples (bool): Treat spatial locations as samples
            sim_t_after_proj (bool): Apply temporal loss after projection
            idm_after_proj (bool): Apply IDM loss after projection
        """
        super().__init__()
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        # Anti-collapse mode: "vicreg" (variance hinge + covariance decorrelation) or
        # "sigreg" (LeJEPA's SIGReg: push the marginal toward an isotropic Gaussian via
        # sliced Epps-Pulley). They are mutually exclusive; sim_t and idm terms are shared.
        self.reg_type = reg_type
        self.sigreg_coeff = sigreg_coeff
        self.sigreg_num_slices = sigreg_num_slices

        # Temporal-distance term: make latent distance grow monotonically with the
        # temporal gap |i-j| along the trajectory, so ||E(s_i)-E(s_j)|| reflects the
        # number of steps in the dataset trajectories (a proxy for geodesic distance,
        # wall-aware) rather than raw Euclidean position distance.
        self.tdist_coeff = tdist_coeff

        # h1/h2 disentangle split (channel dim). If dist_split_frac > 0, the latent is
        # split into h1 = z[:, :k] (k = round(frac * C), the GEOMETRY block carrying the
        # distance/tdist signal) and h2 = z[:, k:] (the CONTROL block carrying IDM). The
        # IDM path stop-grads h1 so action identity only shapes h2; cross_cov_coeff then
        # decorrelates the two blocks. Fraction (not absolute) so it is robust to C.
        self.dist_split_frac = dist_split_frac
        self.cross_cov_coeff = cross_cov_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        # Initialize individual loss components
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x, actions=None):
        """
        Args:
            x: [B, C, T, H, W] - Input activations. Internally reshaped to either
                [1, B, D] when first_t_only=True or [T*B, D] otherwise, with D=C*H*W.
            actions: [B, A, T] - Optional actions for IDM loss
        """
        b, c, t, h, w = x.shape

        # divergent gradient paths for x_unprojected and x_projected
        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [T, B, C*H*W]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, C_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [B, T, H, W, C_out]
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [T, B, C_out*H*W]

        # SIM_T LOSS
        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        # ---- optional h1/h2 disentangle split (along channel dim C) ----
        # h1 = x[:, :k] carries GEOMETRY (distance/tdist); h2 = x[:, k:] carries CONTROL
        # (IDM). k from a fraction so it is robust to the latent channel count.
        k = int(round(self.dist_split_frac * c))
        use_split = self.dist_split_frac > 0.0 and 0 < k < c

        # TEMPORAL-DISTANCE LOSS (rank latent distances by temporal gap |i-j|)
        tdist_loss = torch.zeros((), device=x.device)
        if self.tdist_coeff > 0:
            if use_split:
                x_h1 = x[:, :k].permute(2, 0, 1, 3, 4).reshape(t, b, -1)
                tdist_loss = self._temporal_dist(x_h1)
            else:
                tdist_loss = self._temporal_dist(x_unprojected)

        # IDM LOSS
        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            elif use_split:
                # IDM shapes h2 only: stop-grad on h1 keeps the geometry block free of
                # action-identity info. Input dim stays full C so the IDM module is unchanged.
                x_idm = torch.cat([x[:, :k].detach(), x[:, k:]], dim=1)
                x_idm = x_idm.permute(2, 0, 1, 3, 4).reshape(t, b, -1)
                idm_loss = self.idm_loss_fn(x_idm, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        # CROSS-DECORRELATION between h1 and h2 (squared cross-correlation -> 0)
        cross_cov_loss = torch.zeros((), device=x.device)
        if use_split and self.cross_cov_coeff > 0:
            cross_cov_loss = self._cross_decorr(x, k)

        # STD and COV LOSS
        if self.spatial_as_samples:
            if self.first_t_only:
                # Use only first time: [B*H*W, C_out]
                x_for_vc = x_projected[:, 0].reshape(b * h * w, c_out)
                assert x_for_vc.shape == (b * h * w, c_out)
            else:
                # Use all times: [B*T*H*W, C_out]
                x_for_vc = x_projected.reshape(-1, c_out)
                assert x_for_vc.shape == (b * t * h * w, c_out)
        else:
            x_for_vc = x_projected.permute(0, 1, 4, 2, 3).reshape(
                b, t, -1
            )  # [B, T, C_out*H*W]
            if self.first_t_only:
                # Use only first time: [B, C_out*H*W]
                x_for_vc = x_for_vc[:, 0]
                assert x_for_vc.shape == (b, c_out * h * w)
            else:
                # Use all times: [B*T, C_out*H*W]
                x_for_vc = x_for_vc.reshape(-1, x_for_vc.size(-1))
                assert x_for_vc.shape == (b * t, c_out * h * w)
        # [B*T, C_out*H*W] if first_t_only=False and spatial_as_samples=False
        # or [B, C_out*H*W] if first_t_only=True and spatial_as_samples=False
        # or [B*H*W, C_out] if first_t_only=True spatial_as_samples=True
        # or [B*T*H*W, C_out] if first_t_only=False spatial_as_samples=True
        # ANTI-COLLAPSE LOSS: VICReg (std + cov) or SIGReg (sliced Epps-Pulley)
        zero = torch.zeros((), device=x.device)
        if self.reg_type == "sigreg":
            sigreg_loss = self._sigreg(x_for_vc)
            std_loss = cov_loss = zero
            vc_weighted = self.sigreg_coeff * sigreg_loss
        else:
            std_loss = self.std_loss_fn(x_for_vc)
            cov_loss = self.cov_loss_fn(x_for_vc)
            sigreg_loss = zero
            vc_weighted = self.cov_coeff * cov_loss + self.std_coeff * std_loss

        total_weighted_loss = (
            vc_weighted
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
            + self.tdist_coeff * tdist_loss
            + self.cross_cov_coeff * cross_cov_loss
        )
        total_unweighted_loss = (
            cov_loss + std_loss + sigreg_loss + sim_loss_t + idm_loss
            + tdist_loss + cross_cov_loss
        )

        loss_dict = {
            "cov_loss": cov_loss.item(),
            "std_loss": std_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "sim_loss_t": sim_loss_t.item(),
            "idm_loss": idm_loss if isinstance(idm_loss, float) else idm_loss.item(),
            "tdist_loss": tdist_loss.item(),
            "cross_cov_loss": cross_cov_loss.item(),
        }

        return total_weighted_loss, total_unweighted_loss, loss_dict

    def _cross_decorr(self, x, k):
        """Decorrelate the geometry block h1 = x[:, :k] from the control block h2 = x[:, k:].

        Penalises the mean squared cross-CORRELATION (scale-free) between every h1 feature
        and every h2 feature -> the off-diagonal block of the full covariance matrix. This
        is what stops action-identity info (which IDM injects into h2) from leaking back
        into the distance-bearing block.

        Args:
            x: [B, C, T, H, W] latent activations.
            k: number of channels in h1.
        Returns:
            Scalar cross-decorrelation loss.
        """
        b, c, t, h, w = x.shape
        z = x.permute(0, 2, 3, 4, 1).reshape(-1, c)          # [N, C], N = B*T*H*W
        z = z - z.mean(dim=0, keepdim=True)
        n = z.shape[0]
        a, bb = z[:, :k], z[:, k:]                            # [N, k], [N, C-k]
        cross = (a.transpose(0, 1) @ bb) / (n - 1)           # [k, C-k] cross-covariance
        sa = a.std(dim=0).clamp_min(1e-4)                     # [k]
        sb = bb.std(dim=0).clamp_min(1e-4)                    # [C-k]
        corr = cross / (sa[:, None] * sb[None, :])           # -> cross-correlation
        return corr.pow(2).mean()

    def _temporal_dist(self, x_seq):
        """Make latent distance reflect temporal (trajectory) distance.

        For every anchor a, we want pairs closer in time to be closer in latent space:
        gap(a, p) < gap(a, n)  =>  ||z_a - z_p|| < ||z_a - z_n||. We enforce this with a
        scale-free pairwise logistic (Bradley-Terry) ranking over all such triplets.

        Args:
            x_seq: [T, B, D] encoded trajectory (T time steps, B batch, D features).
        Returns:
            Scalar ranking loss (0 if T < 3).
        """
        T = x_seq.shape[0]
        if T < 3:
            return torch.zeros((), device=x_seq.device)
        z = x_seq.permute(1, 0, 2)                       # [B, T, D]
        d = torch.cdist(z, z, p=2)                        # [B, T, T] latent distances
        idx = torch.arange(T, device=x_seq.device)
        gap = (idx[:, None] - idx[None, :]).abs()         # [T, T] temporal gaps
        valid = gap[:, :, None] < gap[:, None, :]         # [T(a), T(p), T(n)]
        if not valid.any():
            return torch.zeros((), device=x_seq.device)
        d_ap = d[:, :, :, None]                            # [B, T, T(p), 1]
        d_an = d[:, :, None, :]                            # [B, T, 1, T(n)]
        # want d_an - d_ap > 0 ; logistic ranking, averaged over valid triplets & batch
        rank = -F.logsigmoid(d_an - d_ap)                  # [B, T, T, T]
        return rank[:, valid].mean()

    def _sigreg(self, x):
        """SIGReg anti-collapse term (LeJEPA): push the batch marginal toward an
        isotropic N(0, I) using sliced Epps-Pulley statistics.

        Args:
            x: [N, D] batch of embeddings.
        Returns:
            Scalar mean Epps-Pulley statistic over `sigreg_num_slices` random unit slices.
        """
        with torch.no_grad():
            A = torch.randn(x.size(1), self.sigreg_num_slices, device=x.device)
            A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)
        views = x @ A  # [N, num_slices]
        return epps_pulley(views).mean()


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms."""

    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """Compute VICReg loss.

        Args:
            z1: [B, D] - First projection tensor
            z2: [B, D] - Second projection tensor

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


######################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}
