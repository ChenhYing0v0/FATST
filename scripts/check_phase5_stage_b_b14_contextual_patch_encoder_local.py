from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models.TimeAlign import Model  # noqa: E402


def make_config(
    encoder_mode: str,
    patch_len: int = 16,
    stride: int = 8,
    readout_mode: str = "learned-basis-forecast-operator",
) -> SimpleNamespace:
    contextual = encoder_mode == "contextual-patch-transformer"
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        patch_num=48,
        d_model=16 if contextual else 32,
        n_heads=4,
        e_layers=2,
        d_ff=64 if contextual else 32,
        dropout=0.0,
        layer_norm=1,
        pos=1,
        enc_in=3,
        readout_mode=readout_mode,
        encoder_mode=encoder_mode,
        history_patch_len=patch_len,
        history_patch_stride=stride,
        history_attn_dropout=0.0,
        history_res_attention=True,
        basis_rank=256,
        target_horizons=[96, 192, 336, 720],
        local_margin=0.0,
        global_margin=0.0,
        loc=1,
        glo=1,
    )


def legacy_reference(model: Model, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, channels = x.shape
    encoded = model.normalization_x(x, "norm")
    encoded = model.patch_emb_x(
        encoded.permute(0, 2, 1).reshape(-1, channels * seq_len)
    )
    for layer_idx in range(model.e_layers):
        encoded = encoded + model.encoder[layer_idx](encoded)
        if model.layer_norm:
            encoded = model.norm_x[layer_idx](encoded)
    hidden = encoded.reshape(
        batch,
        channels,
        model.patch_num,
        model.d_model,
    ).flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    output = torch.einsum(
        "hk,bck->bch",
        model.learned_temporal_basis,
        coeff,
    ) + model.learned_temporal_bias.view(1, 1, -1)
    return model.normalization_x(output.permute(0, 2, 1), "denorm")


def assert_finite_grad(parameter: torch.nn.Parameter, name: str) -> None:
    if parameter.grad is None:
        raise AssertionError(f"missing gradient for {name}")
    if not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"non-finite gradient for {name}")


def main() -> None:
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    x = torch.randn(2, 720, 3)
    y = torch.randn(2, 720, 3)

    legacy = Model(make_config("timealign-token-mlp")).eval()
    with torch.no_grad():
        actual, recon, align = legacy(x, y, is_training=False, target_prefix=720)
        expected = legacy_reference(legacy, x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert recon is None
    assert float(align) == 0.0
    assert legacy.encode_history(x).shape == (2, 3, 48, 32)

    official = Model(
        make_config(
            "timealign-token-mlp",
            readout_mode="official",
        )
    )
    official.train()
    official_output, official_recon, official_align = official(
        x,
        y,
        is_training=True,
    )
    assert official_output.shape == (2, 720, 3)
    assert official_recon is not None and official_recon.shape == (2, 720, 3)
    assert torch.isfinite(official_output).all()
    assert torch.isfinite(official_recon).all()
    assert torch.isfinite(official_align)

    config_p16 = make_config("contextual-patch-transformer", patch_len=16, stride=8)
    model_p16 = Model(config_p16).train()
    memory_p16 = model_p16.encode_history(x)
    assert memory_p16.shape == (2, 3, 90, 16)
    output_p16, recon, align = model_p16(
        x,
        y,
        is_training=True,
        target_prefix=96,
    )
    assert output_p16.shape == (2, 96, 3)
    assert recon is None
    assert float(align) == 0.0
    assert torch.isfinite(output_p16).all()
    output_p16.square().mean().backward()
    assert_finite_grad(
        model_p16.history_encoder.patch_projection.weight,
        "patch_projection",
    )
    assert_finite_grad(
        model_p16.history_encoder.position_embedding,
        "position_embedding",
    )
    assert_finite_grad(
        model_p16.history_encoder.layers[0].attention.query.weight,
        "attention_query",
    )

    reloaded = Model(config_p16).eval()
    reloaded.load_state_dict(copy.deepcopy(model_p16.state_dict()))
    model_p16.eval()
    with torch.no_grad():
        expected_reload = model_p16(x, y, is_training=False, target_prefix=720)[0]
        actual_reload = reloaded(x, y, is_training=False, target_prefix=720)[0]
    torch.testing.assert_close(actual_reload, expected_reload, rtol=0.0, atol=0.0)

    model_p48 = Model(
        make_config("contextual-patch-transformer", patch_len=48, stride=24)
    ).eval()
    assert model_p48.encode_history(x).shape == (2, 3, 30, 16)

    try:
        Model(
            make_config(
                "contextual-patch-transformer",
                readout_mode="official",
            )
        )
    except ValueError as error:
        assert "requires readout_mode=learned-basis-forecast-operator" in str(error)
    else:
        raise AssertionError("contextual encoder unexpectedly accepted official readout")

    print("legacy_exact_equivalence=pass")
    print("official_alignment_path=pass")
    print("contextual_p16_shape=[2,3,90,16]")
    print("contextual_p48_shape=[2,3,30,16]")
    print("gradient_and_reload=pass")


if __name__ == "__main__":
    main()
