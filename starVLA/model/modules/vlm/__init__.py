import os


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    vlm_key = str(vlm_name).lower()
    if os.path.isdir(vlm_name):
        vlm_key = f"{vlm_key} {os.path.basename(os.path.realpath(vlm_name)).lower()}"

    if "spikebrain" in vlm_key or "spb2_vl" in vlm_key:
        from .SpikeBrain2VL import _SpikeBrain2_VL_Interface
        return _SpikeBrain2_VL_Interface(config)
    elif "qwen2.5-vl" in vlm_key or "nora" in vlm_key: # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif "qwen3-vl" in vlm_key:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    elif "qwen3.5" in vlm_key:
        from .QWen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)
    elif "florence" in vlm_name.lower(): # temp for some ckpt
        from .Florence2 import _Florence_Interface 
        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        from .CosmosReason2 import _CosmosReason2_Interface
        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")


