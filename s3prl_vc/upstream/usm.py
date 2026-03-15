import torch
import torch.nn.functional as F

from transformers import Gemma3nAudioEncoder, Gemma3nAudioFeatureExtractor

class USM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        encoder_id = "Atotti/google-usm"
        source_model_id = "google/gemma-3n-e2b-it"

        self.audio_encoder = Gemma3nAudioEncoder.from_pretrained(encoder_id)
        self.feature_extractor = Gemma3nAudioFeatureExtractor.from_pretrained(source_model_id)

        self.audio_encoder.eval()

        self.model = whisper.load_model(model_size)
        self.hidden_size = self.model.dims.n_audio_state
        self.extend_pe(torch.tensor(0.0).expand(1, default_pe_max_length))

    # required by S3PRL Featurizer
    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    @property
    def num_layers(self):
        return 1

    @property
    def hidden_sizes(self):
        return [self.hidden_size]

    @property
    def downsample_rates(self):
        return [self.get_downsample_rates()]

    def forward(self, speech, speech_lens):
        inputs = self.feature_extractor(
            [speech],
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        audio_mel = inputs["input_features"].to(device)
        audio_mel_mask = (inputs["input_features_mask"] == 0).to(device)

        with torch.inference_mode():

            audio_encodings, output_mask = self.audio_encoder(
                audio_mel=audio_mel,
                audio_mel_mask=audio_mel_mask
            )

        return [audio_encodings], 
