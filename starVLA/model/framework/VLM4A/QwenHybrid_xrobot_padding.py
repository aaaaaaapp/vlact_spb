"""Compatibility alias for the historical ALOHA QwenHybrid framework."""
from starVLA.model.tools import FRAMEWORK_REGISTRY
from .QwenOFT import Qwenvl_OFT

@FRAMEWORK_REGISTRY.register("QwenHybrid_xrobot_padding")
class QwenHybrid_xrobot_padding(Qwenvl_OFT):
    """Historical name mapped to the current QwenOFT implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The 20k ALOHA checkpoint predates the public QwenOFT rename and
        # stores this head below ``oft_action_model``.  Keep that exact module
        # name in the state dict while exposing the modern attribute to the
        # inherited forward methods through the property below.
        head = self._modules.pop("action_model")
        self._modules["oft_action_model"] = head

    @property
    def action_model(self):
        return self.oft_action_model

