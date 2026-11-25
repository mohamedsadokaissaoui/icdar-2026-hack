"""
Focus-picking loss (FocusNet, Zhang, Sheng & Shen, Pattern Recognition 2022 — arXiv:2110.07307).

Stand-alone copy of the loss used in notebooks/circleid_eva02_focusnet.ipynb so it can be
imported, unit-tested and reused with any pair of (frozen baseline, trainable FocusNet) models.

    L = L_cls + alpha * R_attention - beta * R_entropy

    L_cls       cross-entropy on re-weighted logits  z_f + d,  d = softmax(z_f) - softmax(z_b)
                -> samples where FocusNet is LESS confident than the baseline on the true class
                   get a larger loss (hard-sample re-weighting)
    R_attention cross-entropy between softmax(z_f) and the multi-warm label built from z_b
                -> FocusNet must keep probability mass on the classes the baseline finds confusing
    R_entropy   sum p log p  (negative entropy)
                -> subtracting it rewards higher entropy, i.e. discourages over-confidence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_multi_warm_label(baseline_logits: torch.Tensor, ground_truth: torch.Tensor,
                          num_classes: int) -> torch.Tensor:
    """Eq. 4-6: classes with a positive baseline logit + the ground-truth class, rows sum to 1."""
    l_prime = (baseline_logits > 0).float()                                   # Eq. 4
    gt_onehot = F.one_hot(ground_truth, num_classes).float()
    l_double = torch.clamp(l_prime + gt_onehot, max=1.0)                      # Eq. 5
    return l_double / l_double.sum(dim=1, keepdim=True).clamp(min=1.0)        # Eq. 6


class FocusPickingLoss(nn.Module):
    """Eq. 12. `baseline_logits` must come from a frozen model (no gradient is taken through it)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, focusnet_logits, baseline_logits, ground_truth):
        C = focusnet_logits.shape[1]
        baseline_logits = baseline_logits.detach().float()
        focusnet_logits = focusnet_logits.float()

        multi_warm = make_multi_warm_label(baseline_logits, ground_truth, C)
        p_f = F.softmax(focusnet_logits, dim=1)
        p_b = F.softmax(baseline_logits, dim=1)
        d = p_f.detach() - p_b                                                 # difficulty signal

        L_cls = F.cross_entropy(focusnet_logits + d, ground_truth)             # Eq. 7-9
        R_att = -(multi_warm * torch.log(p_f + self.eps)).sum(1).mean()        # Eq. 10
        R_ent = (p_f * torch.log(p_f + self.eps)).sum(1).mean()                # Eq. 11
        return L_cls + self.alpha * R_att - self.beta * R_ent


def mixup_focus_loss(criterion, focus_logits, baseline_logits, y_a, y_b, lam):
    """MixUp / CutMix: blend the loss for both label sets, as in the notebook."""
    return lam * criterion(focus_logits, baseline_logits, y_a) + \
        (1 - lam) * criterion(focus_logits, baseline_logits, y_b)


if __name__ == "__main__":
    # sanity check on a toy 8-pen batch
    torch.manual_seed(0)
    zb = torch.tensor([[-1.0, -1.2, 2.1, -0.4, -1.5, -0.9, 2.8, 0.6]])        # baseline: 3, 7, 8 positive
    y = torch.tensor([6])                                                     # true pen = 7
    print("multi-warm:", make_multi_warm_label(zb, y, 8).numpy().round(3))  # 1/3 on pens 3, 7, 8

    crit = FocusPickingLoss()
    zf = torch.randn(4, 8, requires_grad=True)
    zb4 = torch.randn(4, 8)
    loss = crit(zf, zb4, torch.tensor([0, 2, 6, 7]))
    loss.backward()
    assert torch.isfinite(loss) and zf.grad is not None
    print(f"loss = {loss.item():.4f}  (grad ok)")
