import torch
from s3prl.nn import S3PRLUpstream
from s3prl_vc.upstream.ppg_sxliu.model import build_ppg_model

def get_upstream(name: str) -> torch.nn.Module:
    if name in S3PRLUpstream.available_names():
        return S3PRLUpstream(name)
    elif name == "ppg_sxliu":
        return build_ppg_model()
    else:
        raise ValueError("upstream not supported.")
        exit(1)