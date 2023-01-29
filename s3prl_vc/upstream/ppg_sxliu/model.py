import argparse
import torch
from pathlib import Path
import yaml


from s3prl_vc.upstream.ppg_sxliu.frontend import DefaultFrontend
from s3prl_vc.upstream.ppg_sxliu.utterance_mvn import UtteranceMVN
from s3prl_vc.upstream.ppg_sxliu.encoder.conformer_encoder import ConformerEncoder
from s3prl_vc.utils.download import _urls_to_filepaths

TRAIN_CONFIG_URL="https://github.com/liusongxiang/ppg-vc/raw/main/conformer_ppg_model/en_conformer_ctc_att/config.yaml"
MODEL_FILE_URL="https://github.com/liusongxiang/ppg-vc/raw/main/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth"


class PPGModel(torch.nn.Module):
    def __init__(
        self,
        frontend,
        normalizer,
        encoder,
    ):
        super().__init__()
        self.frontend = frontend
        self.normalize = normalizer
        self.encoder = encoder
        self.hidden_size = encoder.output_size()
    
    # required by S3PRL Featurizer
    def get_downsample_rates(self, key: str=None) -> int:
        return 160

    @property
    def num_layers(self):
        return 1

    @property
    def hidden_sizes(self):
        return [self.hidden_size]

    @property
    def downsample_rates(self):
        return [self.get_downsample_rates()]

    def forward(self, speech, speech_lengths):
        """

        Args:
            speech (tensor): (B, L)
            speech_lengths (tensor): (B, )

        Returns:
            bottle_neck_feats (tensor): (B, L//hop_size, 144)

        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        feats, feats_lengths = self.normalize(feats, feats_lengths)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # As required by S3PRL Featurizer, needs to be returned in lists
        return [encoder_out], [encoder_out_lens]

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ):
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths
        

def build_model(args):
    normalizer = UtteranceMVN(**args.normalize_conf)
    frontend = DefaultFrontend(**args.frontend_conf)
    encoder = ConformerEncoder(input_size=80, **args.encoder_conf)
    model = PPGModel(frontend, normalizer, encoder)
    
    return model


def build_ppg_model():

    # download from Songxiang's repo
    train_config = _urls_to_filepaths(TRAIN_CONFIG_URL)
    model_file = _urls_to_filepaths(MODEL_FILE_URL)

    config_file = Path(train_config)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args = argparse.Namespace(**args)

    model = build_model(args)
    model_state_dict = model.state_dict()

    ckpt_state_dict = torch.load(model_file, map_location='cpu')
    ckpt_state_dict = {k:v for k,v in ckpt_state_dict.items() if 'encoder' in k}

    model_state_dict.update(ckpt_state_dict)
    model.load_state_dict(model_state_dict)

    return model