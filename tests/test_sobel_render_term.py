"""w_sobel 이 edge_gain 과 다른 것을 말하고 있는지 고정한다.

핵심 성질은 하나다: 평균은 맞고 대비만 잃은 렌더를, 가중 L1 은 거의 벌하지 않지만
이 항은 벌한다. 이게 성립하지 않으면 항을 넣을 이유가 없다.
"""
import torch
import torch.nn.functional as F

from can3tok.render import photometric_loss, _sobel


def _edge_image(h=64, w=64):
    """왼쪽은 어둡고 오른쪽은 밝은, 경계가 하나 있는 그림.

    0/1 이 아니라 0.25/0.75 인 이유: 아래 테스트가 전체에 상수를 더한 그림과 비교하는데,
    포화되면 그 그림의 L1 이 의도한 값과 달라져 비교가 무너진다.
    """
    im = torch.full((3, h, w), 0.25)
    im[:, :, w // 2:] = 0.75
    return im


def _blurred(im, k=9):
    r = k // 2
    g = torch.ones(1, 1, 1, k) / k
    a = F.pad(im[None], (r, r, 0, 0), mode="replicate")
    return F.conv2d(a, g.expand(3, 1, 1, k), groups=3)[0]


def _blur_and_equal_l1_shift(ref):
    """흐린 그림과, 그것과 픽셀 L1 이 정확히 같은 '상수 더한' 그림.

    상수를 더하면 기울기는 기준과 완전히 같으므로 Sobel 항은 0 이다. 즉 두 그림은
    L1 로는 구별되지 않고 Sobel 항으로만 구별된다. 이게 이 항의 존재 이유다.
    """
    blur = _blurred(ref)
    shift = ref + float((blur - ref).abs().mean())
    assert float(shift.max()) <= 1.0
    return blur, shift


def test_plain_l1_cannot_tell_blur_from_a_uniform_brightness_error():
    """문제의 출발점. 두 그림의 L1 은 같다 -- L1 만 보면 흐림을 고를 이유가 없다."""
    ref = _edge_image()
    blur, shift = _blur_and_equal_l1_shift(ref)
    a, _, _ = photometric_loss(blur, ref, lam_dssim=0.0, edge_gain=0.0)
    b, _, _ = photometric_loss(shift, ref, lam_dssim=0.0, edge_gain=0.0)
    assert abs(float(a) - float(b)) < 1e-6, (float(a), float(b))


def test_edge_gain_does_separate_them_on_a_lone_edge():
    """edge_gain 은 무력하지 않다. 경계가 하나뿐인 그림에서는 10배 가까이 가른다.

    이 테스트가 여기 있는 이유는 반대 주장을 막기 위해서다. w_sobel 의 근거는
    "edge_gain 이 흐림을 구별하지 못한다" 가 아니다 -- 구별한다. 근거는 실제 장면에서
    기준 이미지의 엣지 질량이 모델이 이미 맞히는 실루엣과 창틀에 몰려 있어서,
    가중치가 '빠진 엣지' 가 아니라 '엣지 일반' 으로 예산을 옮긴다는 것이다. 그 주장은
    합성 이미지로는 확인할 수 없고 실제 렌더 프로브로만 확인된다.
    """
    ref = _edge_image()
    blur, shift = _blur_and_equal_l1_shift(ref)
    a, _, _ = photometric_loss(blur, ref, lam_dssim=0.0, edge_gain=4.0)
    b, _, _ = photometric_loss(shift, ref, lam_dssim=0.0, edge_gain=4.0)
    assert float(a) > 5.0 * float(b), (float(a), float(b))


def test_sobel_term_separates_them():
    """같은 L1 을 가진 두 그림 중 흐린 쪽을 분명히 더 나쁘게 본다."""
    ref = _edge_image()
    blur, shift = _blur_and_equal_l1_shift(ref)
    a, _, _ = photometric_loss(blur, ref, lam_dssim=0.0, edge_gain=4.0, w_sobel=1.0)
    b, _, _ = photometric_loss(shift, ref, lam_dssim=0.0, edge_gain=4.0, w_sobel=1.0)
    assert float(a) > 2.0 * float(b), (float(a), float(b))


def test_exact_match_is_still_zero():
    ref = _edge_image()
    v, _, _ = photometric_loss(ref, ref, lam_dssim=0.0, edge_gain=4.0, w_sobel=1.0)
    assert float(v) < 1e-6


def test_zero_weight_changes_nothing():
    ref = _edge_image()
    blur = _blurred(ref)
    a, _, _ = photometric_loss(blur, ref, lam_dssim=0.2, edge_gain=4.0, w_sobel=0.0)
    b, _, _ = photometric_loss(blur, ref, lam_dssim=0.2, edge_gain=4.0)
    assert abs(float(a) - float(b)) < 1e-9


def test_gradient_reaches_the_prediction():
    ref = _edge_image()
    pred = _blurred(ref).clone().requires_grad_(True)
    v, _, _ = photometric_loss(pred, ref, lam_dssim=0.0, edge_gain=0.0, w_sobel=1.0)
    v.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


def test_sobel_part_adds_no_gradient_to_the_reference():
    """이 항이 기준 이미지를 끌어당기면 목표가 같이 흐려진다. L1 항의 기존 동작은
    그대로 두고, 추가된 부분만 기여가 없는지 본다 (실사용에서 기준은 no_grad 로 만든다)."""
    pred = _blurred(_edge_image())
    grads = []
    for w in (0.0, 1.0):
        ref = _edge_image().clone().requires_grad_(True)
        v, _, _ = photometric_loss(pred, ref, lam_dssim=0.0, edge_gain=0.0, w_sobel=w)
        v.backward()
        grads.append(0.0 if ref.grad is None else float(ref.grad.abs().sum()))
    assert abs(grads[0] - grads[1]) < 1e-6, grads


def test_sobel_shape():
    im = _edge_image(32, 48)
    assert tuple(_sobel(im).shape) == (6, 32, 48)


def test_detail_gains_ramps_the_sobel_weight():
    from types import SimpleNamespace
    from can3tok.schedule import detail_gains
    a = SimpleNamespace(detail_start=1000, detail_phase_ramp=100, edge_gain=4.0,
                        w_render_perc=0.05, w_sobel=1.0)
    assert detail_gains(999, a) == (0.0, 0.0, 0.0)
    mid = detail_gains(1050, a)
    assert 0.0 < mid[2] < 1.0
    assert detail_gains(2000, a)[2] == 1.0
