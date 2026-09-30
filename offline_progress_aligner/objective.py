"""Original self-supervised objective; no cross-video frame labels are used."""
import torch


def cycle_direction(a, b, source_positions, cfg, dtw):
    sim = -torch.cdist(a, b).square() / cfg["similarity_temperature"]
    beta, path = dtw(sim, gamma_s=cfg["dtw_gamma"], gamma_f=cfg["dtw_column_temperature"],
                     bidirectional=True, method="anti_diag", return_cost=True)
    soft_matches = beta @ b
    cycle = (-torch.cdist(soft_matches, a).square() / cfg["similarity_temperature"]).softmax(-1)
    positions = source_positions[:, None, :]
    mean = (cycle * positions).sum(-1)
    variance = (cycle * (positions - mean[..., None]).square()).sum(-1).clamp_min(1e-4)
    loss = ((mean - source_positions).square() / variance
            + cfg["cycle_variance_weight"] * variance.log()).mean()
    return loss, (path / (a.shape[1] + b.shape[1])).mean(), (mean - source_positions).abs().mean()


def objective(a, b, pa, pb, cfg, dtw):
    la, ca, ea = cycle_direction(a, b, pa, cfg, dtw)
    lb, cb, eb = cycle_direction(b, a, pb, cfg, dtw)
    cycle, path = (la + lb) / 2, (ca + cb) / 2
    total = cycle + cfg["path_loss_weight"] * path
    return total, {"loss": float(total.detach()), "cycle_loss": float(cycle.detach()),
                   "path_loss": float(path.detach()), "cycle_return_mae": float(((ea + eb) / 2).detach())}
