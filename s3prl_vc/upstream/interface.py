import torch
from s3prl.nn import S3PRLUpstream

def get_upstream(name: str) -> torch.nn.Module:
    if name in S3PRLUpstream.available_names():
        return S3PRLUpstream(name)
    elif name == "ppg_sxliu":
        from s3prl_vc.upstream.ppg_sxliu.model import build_ppg_model
        return build_ppg_model()
    elif name == "ppg_whisper":
        from s3prl_vc.upstream.whisper import WhisperPPG
        return WhisperPPG()
    else:
        raise ValueError("upstream not supported.")
        exit(1)