"""aniso_shape_loss 가 주장대로 동작하는지. 특히 '구에서도 기울기가 산다'는 부분."""
import torch

from can3tok.losses import aniso_shape_loss, covariance3d_loss

LAYOUT = {"scale": 3, "rot": 6, "opacity": 10, "color": 11, "sh": 14}


def _g(n, ls, quat):
    t = torch.zeros(1, n, 14)
    t[..., 3:6] = torch.as_tensor(ls, dtype=torch.float32)
    t[..., 6:10] = torch.as_tensor(quat, dtype=torch.float32)
    return t


def test_ignores_rotation():
    """회전은 이 항의 관심사가 아니다. 크기와 방향을 나눠 실으려는 것이 목적."""
    m = torch.ones(1, 8)
    tgt = _g(8, [-2.0, -5.0, -6.0], [1, 0, 0, 0])
    a = _g(8, [-3.0, -4.0, -5.0], [1, 0, 0, 0])
    b = _g(8, [-3.0, -4.0, -5.0], [0.7071, 0.7071, 0, 0])
    la = aniso_shape_loss(a, tgt, m, LAYOUT)
    lb = aniso_shape_loss(b, tgt, m, LAYOUT)
    assert torch.allclose(la, lb), (la, lb)


def test_invariant_to_axis_permutation():
    """(R, s) 분해의 축 치환 모호성이 값을 바꾸면 안 된다."""
    m = torch.ones(1, 4)
    tgt = _g(4, [-2.0, -5.0, -6.0], [1, 0, 0, 0])
    a = _g(4, [-3.0, -4.0, -7.0], [1, 0, 0, 0])
    b = _g(4, [-7.0, -3.0, -4.0], [1, 0, 0, 0])
    assert torch.allclose(aniso_shape_loss(a, tgt, m, LAYOUT),
                          aniso_shape_loss(b, tgt, m, LAYOUT))


def test_ignores_overall_size():
    """전체 크기는 covariance 항의 몫이다. 여기서는 비율만 본다."""
    m = torch.ones(1, 4)
    tgt = _g(4, [-2.0, -5.0, -6.0], [1, 0, 0, 0])
    a = _g(4, [-3.0, -6.0, -7.0], [1, 0, 0, 0])          # tgt 를 통째로 -1
    assert float(aniso_shape_loss(a, tgt, m, LAYOUT)) < 1e-6


def test_gradient_survives_on_a_sphere():
    """이 항이 존재하는 이유. 구면 예측에서 covariance 항의 회전 기울기는 0 이고,
    scale 기울기도 크기 불일치에 눌린다. 이 항은 구에서도 늘리는 방향을 가리켜야 한다."""
    m = torch.ones(1, 64)
    tgt = _g(64, [-2.0, -6.0, -6.5], [1, 0, 0, 0])       # 바늘
    sph = _g(64, [-4.0, -4.0, -4.0], [1, 0, 0, 0])       # 구
    p = sph.clone().requires_grad_(True)
    aniso_shape_loss(p, tgt, m, LAYOUT).backward()
    g = p.grad[..., 3:6]
    assert torch.isfinite(g).all()
    assert g.abs().max() > 1e-3, g.abs().max()
    # 가장 큰 축은 키우고(음의 기울기) 가장 작은 축은 줄이는 방향이어야 한다.
    assert g[0, 0, 0] < 0 < g[0, 0, 2], g[0, 0]


def test_covariance_term_is_blind_to_rotation_on_a_sphere():
    """대조: 같은 구면 예측에서 회전을 바꿔도 covariance 항은 꿈쩍하지 않는다."""
    m = torch.ones(1, 16)
    w = {"w_cov3d": 1.0}
    tgt = _g(16, [-2.0, -6.0, -6.5], [1, 0, 0, 0])
    a = _g(16, [-4.0, -4.0, -4.0], [1, 0, 0, 0])
    b = _g(16, [-4.0, -4.0, -4.0], [0.7071, 0, 0.7071, 0])
    ca = covariance3d_loss(a, tgt, m, LAYOUT, w)
    cb = covariance3d_loss(b, tgt, m, LAYOUT, w)
    assert torch.allclose(ca, cb, rtol=1e-5), (ca, cb)
