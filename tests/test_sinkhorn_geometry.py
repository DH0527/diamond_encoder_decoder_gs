import torch

from can3tok.losses import group_centroid_loss, intra_group_sinkhorn_loss


def test_sinkhorn_is_permutation_invariant_and_detects_duplicates():
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    mask = torch.ones(1, 4)
    permuted = target[:, [2, 0, 3, 1]].clone().requires_grad_(True)
    duplicate = target[:, [0, 0, 2, 3]].clone()
    good = intra_group_sinkhorn_loss(
        permuted, target, mask, 4, epsilon=0.01, iterations=20, chunk_groups=1
    )
    bad = intra_group_sinkhorn_loss(
        duplicate, target, mask, 4, epsilon=0.01, iterations=20, chunk_groups=1
    )
    assert good.item() < 1e-3
    assert bad.item() > good.item() + 0.1
    good.backward()
    assert torch.isfinite(permuted.grad).all()


def test_sinkhorn_ignores_padding_and_centroid_loss_sees_shift():
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]]]
    )
    pred = target.clone()
    pred[:, 2:] = -123.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss = intra_group_sinkhorn_loss(
        pred, target, mask, 4, epsilon=0.01, iterations=20, chunk_groups=1
    )
    assert loss.item() < 1e-3

    shifted = pred.clone()
    shifted[:, :2] += torch.tensor([0.5, 0.0, 0.0])
    assert group_centroid_loss(shifted, target, mask, 4).item() > 0.49
