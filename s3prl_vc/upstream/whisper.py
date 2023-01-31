import torch
import whisper
import torch.nn.functional as F

class WhisperPPG(torch.nn.Module):
    def __init__(self, model_size: str = "small", default_pe_max_length: int = 5000):
        super().__init__()
        self.model = whisper.load_model(model_size)
        self.hidden_size = self.model.dims.n_audio_state
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, default_pe_max_length))

    def extend_pe(self, x):
        if self.pe is not None:
            if x.shape[1] <= self.pe.shape[1]:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe = whisper.model.sinusoids(x.shape[1], self.hidden_size)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    # required by S3PRL Featurizer
    def get_downsample_rates(self, key: str=None) -> int:
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
        x = whisper.audio.log_mel_spectrogram(speech) # [B, 80, L]

        x = F.gelu(self.model.encoder.conv1(x))
        x = F.gelu(self.model.encoder.conv2(x))
        x = x.permute(0, 2, 1)
        
        self.extend_pe(x)
        x = (x + self.pe[:, :x.shape[1]])

        for block in self.model.encoder.blocks:
            x = block(x)

        x = self.model.encoder.ln_post(x)
        return [x], [torch.round(speech_lens / self.get_downsample_rates())]