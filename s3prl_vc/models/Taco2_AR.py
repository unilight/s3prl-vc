import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

################################################################################

# The follow section is related to Tacotron2
# Reference: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2


def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Taco2Encoder(torch.nn.Module):
    """Encoder module of the Tacotron2 TTS model.

    Reference:
    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Taco2Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        self.input_layer = torch.nn.Linear(idim, econv_chans)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = econv_chans
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the padded acoustic feature sequence (B, Lmax, idim)
        """
        xs = self.input_layer(xs).transpose(1, 2)

        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(
            xs.transpose(1, 2), ilens.cpu(), batch_first=True, enforce_sorted=False
        )
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Lmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens


class Taco2Prenet(torch.nn.Module):
    """Prenet module for decoder of Tacotron2.

    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps alleviate the exposure bias problem.

    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by
        Conditioning WaveNet on Mel Spectrogram Predictions`_.

    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        super(Taco2Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()
        for layer in range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet += [
                torch.nn.Sequential(torch.nn.Linear(n_inputs, n_units), torch.nn.ReLU())
            ]

    def forward(self, x):
        # Make sure at least one dropout is applied.
        if len(self.prenet) == 0:
            return F.dropout(x, self.dropout_rate)

        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x


################################################################################


class RNNLayer(nn.Module):
    """RNN wrapper, includes time-downsampling"""

    def __init__(
        self,
        input_dim,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        sample_rate,
        proj,
    ):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True
        )

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):

        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(
            input_x, x_len, batch_first=True, enforce_sorted=False
        )
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, "drop")

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class RNNCell(nn.Module):
    """RNN cell wrapper"""

    def __init__(self, input_dim, module, dim, dropout, layer_norm, proj):
        super(RNNCell, self).__init__()
        # Setup
        rnn_out_dim = dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = proj

        # Recurrent cell
        self.cell = getattr(nn, module.upper() + "Cell")(input_dim, dim)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, z, c):

        # Forward RNN cell
        new_z, new_c = self.cell(input_x, (z, c))

        # Normalizations
        if self.layer_norm:
            new_z = self.ln(new_z)
        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c


################################################################################


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m


################################################################################


class Taco2_AR(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        resample_ratio,
        stats,
        # model params below
        ar,
        encoder_type,
        hidden_dim,
        lstmp_layers,
        lstmp_dropout_rate,
        lstmp_proj_dim,
        lstmp_layernorm,
        prenet_layers=2,
        prenet_dim=256,
        prenet_dropout_rate=0.5,
        use_f0=False,
        f0_emb_dim=256,
        f0_emb_integration_type="add",
        f0_quantize=True,
        f0_bins=256,
        f0_min=0,
        f0_max=1,
        f0_gaussian_blur=False,
        f0_emb_kernel_size=9,
        f0_emb_dropout=0.5,
        **kwargs
    ):
        super(Taco2_AR, self).__init__()

        self.ar = ar
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.resample_ratio = resample_ratio
        self.use_f0 = use_f0
        self.f0_quantize = f0_quantize
        self.f0_gaussian_blur = f0_gaussian_blur
        self.f0_emb_integration_type = f0_emb_integration_type

        self.register_buffer("target_mean", stats["mean"].float())
        self.register_buffer("target_scale", stats["scale"].float())

        # define encoder
        if encoder_type == "taco2":
            self.encoder = Taco2Encoder(input_dim, eunits=hidden_dim)
        elif encoder_type == "ffn":
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()
            )
        else:
            raise ValueError("Encoder type not supported.")

        # define prenet
        self.prenet = Taco2Prenet(
            idim=output_dim,
            n_layers=prenet_layers,
            n_units=prenet_dim,
            dropout_rate=prenet_dropout_rate,
        )

        # define decoder (LSTMPs)
        self.lstmps = nn.ModuleList()
        for i in range(lstmp_layers):
            if ar:
                prev_dim = output_dim if prenet_layers == 0 else prenet_dim
                rnn_input_dim = hidden_dim + prev_dim if i == 0 else hidden_dim
                rnn_layer = RNNCell(
                    rnn_input_dim,
                    "LSTM",
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    proj=True,
                )
            else:
                rnn_input_dim = hidden_dim
                rnn_layer = RNNLayer(
                    rnn_input_dim,
                    "LSTM",
                    False,
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    sample_rate=1,
                    proj=True,
                )
            self.lstmps.append(rnn_layer)

        # projection layer
        self.proj = torch.nn.Linear(hidden_dim, output_dim)

        # f0 related
        if use_f0:
            if f0_quantize:
                self.f0_bins = torch.linspace(f0_min, f0_max, f0_bins)
                self.f0_emb = Embedding(f0_bins, f0_emb_dim)
            else:
                self.f0_emb = torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels=1,
                        out_channels=f0_emb_dim,
                        kernel_size=f0_emb_kernel_size,
                        padding=(f0_emb_kernel_size - 1) // 2,
                    ),
                    torch.nn.Dropout(f0_emb_dropout),
                )

            # define projection layer for integration
            if self.f0_emb_integration_type == "add":
                self.f0_emb_projection = torch.nn.Linear(f0_emb_dim, hidden_dim)
            elif self.f0_emb_integration_type == "concat":
                self.f0_emb_projection = torch.nn.Linear(
                    hidden_dim + f0_emb_dim, hidden_dim
                )
            else:
                raise ValueError("f0_emb_integration_type not supported.")

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale

    def forward(self, features, lens, f0s=None, targets=None, spk_embs=None):
        """Calculate forward propagation.
        Args:
        features: Batch of the sequences of input features (B, Lmax, idim).
        targets: Batch of the sequences of padded target features (B, Lmax, odim).
        """
        B = features.shape[0]

        # resample the input features according to resample_ratio
        features = features.permute(0, 2, 1)
        resampled_features = F.interpolate(features, scale_factor=self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # encoder
        if self.encoder_type == "taco2":
            encoder_states, lens = self.encoder(
                resampled_features, lens
            )  # (B, Lmax, hidden_dim)
        elif self.encoder_type == "ffn":
            encoder_states = self.encoder(resampled_features)  # (B, Lmax, hidden_dim)

        if self.use_f0:
            assert f0s is not None
            if self.f0_quantize:
                raise NotImplementedError
            else:
                f0_embs = self.f0_emb(f0s.unsqueeze(1)).transpose(
                    1, 2
                )  # (B, Lmax, hidden_dim)
            encoder_states, lens = self._integrate_with_emb(
                encoder_states,
                lens,
                f0_embs,
                self.f0_emb_integration_type,
                self.f0_emb_projection,
            )

        # if the length of `encoder_states` is longer than that of `targets`, match to that of `targets`.
        if targets is not None and targets.shape[1] < encoder_states.shape[1]:
            encoder_states = encoder_states[:, : targets.shape[1]]
            for i in range(lens.shape[0]):
                if lens[i] > targets.shape[1]:
                    lens[i] = targets.shape[1]

        # decoder: LSTMP layers & projection
        if self.ar:
            if targets is not None:
                targets = targets.transpose(0, 1)  # (Lmax, B, output_dim)
            predicted_list = []

            # initialize hidden states
            c_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            z_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            for _ in range(1, len(self.lstmps)):
                c_list += [encoder_states.new_zeros(B, self.hidden_dim)]
                z_list += [encoder_states.new_zeros(B, self.hidden_dim)]
            prev_out = encoder_states.new_zeros(B, self.output_dim)

            # step-by-step loop for autoregressive decoding
            for t, encoder_state in enumerate(encoder_states.transpose(0, 1)):
                concat = torch.cat(
                    [encoder_state, self.prenet(prev_out)], dim=1
                )  # each encoder_state has shape (B, hidden_dim)
                for i, lstmp in enumerate(self.lstmps):
                    lstmp_input = concat if i == 0 else z_list[i - 1]
                    z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
                predicted_list += [
                    self.proj(z_list[-1]).view(B, self.output_dim, -1)
                ]  # projection is done here to ensure output dim
                prev_out = (
                    targets[t]
                    if targets is not None
                    else predicted_list[-1].squeeze(-1)
                )  # targets not None = teacher-forcing
                prev_out = self.normalize(prev_out)  # apply normalization
            predicted = torch.cat(predicted_list, dim=2)
            predicted = predicted.transpose(
                1, 2
            )  # (B, hidden_dim, Lmax) -> (B, Lmax, hidden_dim)
        else:
            predicted = encoder_states
            for i, lstmp in enumerate(self.lstmps):
                predicted, lens = lstmp(predicted, lens)

            # projection layer
            predicted = self.proj(predicted)

        return predicted, lens

    def _integrate_with_emb(self, hs, lens, embs, type, emb_projection):
        """Integrate speaker/f0 embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, Lmax, hdim).
            lens (Tensor): Batch of lengths of the hidden state sequences (B).
            embs (Tensor): Batch of speaker/f0 embeddings (B, embed_dim) or (B, Lmax, embed_dim).
            type (string): "add" or "concat"
            projection (nn.Module)
        """
        # length adjustment
        if len(embs.shape) > 2:
            if embs.shape[1] > hs.shape[1]:
                embs = embs[:, : hs.shape[1]]
            if hs.shape[1] > embs.shape[1]:
                hs = hs[:, : embs.shape[1]]
                lens = torch.where(
                    lens > embs.shape[1], embs.shape[1], lens
                )  # NOTE(unilight): modify lens if hs is also modified
        else:
            embs = embs.unsqueeze(1)

        if type == "add":
            # apply projection and then add to hidden states
            embs = emb_projection(F.normalize(embs, dim=-1))
            hs = hs + embs
        elif type == "concat":
            # concat hidden states with spk embeds and then apply projection
            embs = F.normalize(embs, dim=-1).expand(-1, hs.size(1), -1)
            hs = emb_projection(torch.cat([hs, embs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs, lens
