from types import SimpleNamespace

from PIL import Image
import torch
import torch.nn as nn

from starVLA.model.modules.vlm.SpikeBrain2VL import (
    _SpikeBrain2_VL_Interface,
    _cast_all_floating_state_to_bf16,
    _image_pixel_budget,
)


def test_spikebrain_uses_vlact_224_resolution_as_pixel_budget():
    images = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]

    assert _image_pixel_budget(images) == {
        "min_pixels": 224 * 224,
        "max_pixels": 224 * 224,
    }


def test_spikebrain_pixel_budget_tracks_dynamic_vlact_bucket():
    images = [Image.new("RGB", (320, 192)), Image.new("RGB", (256, 256))]

    assert _image_pixel_budget(images) == {
        "min_pixels": 320 * 192,
        "max_pixels": 256 * 256,
    }


def test_spikebrain_casts_gdn_and_norm_state_to_bf16():
    model = nn.Module()
    model.gdn_state = nn.Parameter(torch.ones(2, dtype=torch.float32))
    model.norm = nn.LayerNorm(2, dtype=torch.float32)
    model.register_buffer("running", torch.ones(2, dtype=torch.float32))

    _cast_all_floating_state_to_bf16(model)

    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(buffer.dtype == torch.bfloat16 for buffer in model.buffers())


def test_spikebrain_backbone_forward_preserves_hidden_state_contract():
    class _Backbone(nn.Module):
        def forward(self, input_ids=None, **kwargs):
            first = torch.ones((*input_ids.shape, 4))
            last = first * 2
            return SimpleNamespace(
                last_hidden_state=last,
                hidden_states=(first, last),
                past_key_values=None,
                attentions=None,
            )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Backbone()

    interface = _SpikeBrain2_VL_Interface.__new__(_SpikeBrain2_VL_Interface)
    nn.Module.__init__(interface)
    interface.model = _Model()
    interface._captured_hidden = None

    output = interface(input_ids=torch.tensor([[1, 2, 3]]))

    assert len(output.hidden_states) == 2
    assert torch.equal(output.last_hidden_state, output.hidden_states[-1])
    assert output.logits is None
