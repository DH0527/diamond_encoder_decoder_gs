from argparse import Namespace

from can3tok.schedule import lr_scales, phase_of, schedule_flags


def _args(**overrides):
    values = dict(
        latent_end=0,
        geo_end=1,
        gen_end=999_999_999,
        gen_start=999_999_999,
        polish_start=1,
        encoder_warmup_steps=0,
        encoder_warmup_decoder_lr_scale=1.0,
        encoder_warmup_gen_lr_scale=1.0,
        late_codec_start=999_999_999,
        late_codec_lr_scale=1.0,
        polish_encoder_scale=0.10,
        polish_attr_encoder_scale=0.20,
        polish_compressor_scale=0.20,
        polish_decompressor_scale=0.20,
        polish_decoder_scale=0.50,
        polish_attr_scale=1.0,
        attr_start=0,
        attr_force_steps=0,
        attr_anneal_steps=1,
        stage="geometry",
        no_gen_branch=True,
        encoder_residual=True,
        decoder_refine_start=0,
        decoder_refine_ramp_steps=1,
        folding_res_start=0,
        folding_res_ramp_steps=1,
        folding_res_cap=1.0,
        encoder_residual_start=0,
        attr_detach_geometry=1,
        attr_detach_release=-1,
        shortcut_alpha_init=0.0,
        shortcut_alpha_final=0.0,
        shortcut_alpha_start=999_999_999,
        shortcut_alpha_ramp_steps=1,
        render_downscale=2,
        render_downscale_start=0,
        render_downscale_steps=1,
    )
    values.update(overrides)
    return Namespace(**values)


def test_polish_overrides_unfinished_gen_curriculum():
    args = _args()
    assert phase_of(0, args) == "geo"
    assert phase_of(1, args) == "polish"
    assert lr_scales(1, args) == {
        "encoder": 0.10,
        "attr_enc": 0.20,
        "compressor": 0.20,
        "decompressor": 0.20,
        "decoder": 0.50,
        "gen": 1.0,
        "attr": 1.0,
        "joint": 1.0,
    }


def test_finetune_teacher_forcing_is_off_after_initial_step():
    args = _args()
    assert schedule_flags(0, args)["attr_teacher_prob"] == 1.0
    assert schedule_flags(1, args)["attr_teacher_prob"] == 0.0


def test_zero_polish_start_still_disables_polish():
    args = _args(polish_start=0)
    assert phase_of(10, args) == "joint"
