# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules based on kaldi-style scp files."""

import logging

from multiprocessing import Manager

import kaldiio
import librosa
import numpy as np
import soundfile as sf

import torch
from torch.utils.data import Dataset

from s3prl_vc.transform.spectrogram import logmelfilterbank
from s3prl_vc.transform.f0 import get_yaapt_f0, get_world_f0
from s3prl_vc.utils import find_files, get_basename, read_hdf5


class MelDataset(Dataset):
    def __init__(
        self,
        config,
        extract_f0=False,
        f0_extractor="yaapt",
        f0_min=None,
        f0_max=None,
        log_f0=True,
        f0_normalize=False,
        f0_mean=None,
        f0_scale=None,
        use_spk_emb=False,
        spk_emb_extractor="wespeaker",
        spk_emb_source="self",  # set to self during training, set to external during inference
        return_utt_id=False,
        return_sampling_rate=False,
        return_wavpath=False,
        allow_cache=False,
        *args,
        **kwargs,
    ):

        # f0 related
        self.extract_f0 = extract_f0
        self.f0_extractor = f0_extractor
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.log_f0 = log_f0
        self.f0_normalize = f0_normalize
        self.f0_mean = f0_mean
        self.f0_scale = f0_scale
        self.spk_emb_source = spk_emb_source

        # speaker embedding related
        self.use_spk_emb = use_spk_emb
        self.spk_emb_source = spk_emb_source

        self.config = config
        self.return_utt_id = return_utt_id
        self.return_sampling_rate = return_sampling_rate
        self.return_wavpath = return_wavpath
        self.allow_cache = allow_cache

        # only load speaker embedding model when testing (i.e. spk_emb_source != self)
        if use_spk_emb and spk_emb_source != "self":
            if spk_emb_extractor == "wespeaker":
                from s3prl_vc.utils.speaker_embedding_wespeaker import (
                    load_asv_model,
                    get_embedding,
                )
                self.spk_emb_model = load_asv_model()
                self.spk_emb_func = get_embedding
            elif spk_emb_extractor == "resemblyzer":
                from s3prl_vc.utils.speaker_embedding_resemblyzer import (
                    load_asv_model,
                    get_embedding,
                )
                self.spk_emb_model = load_asv_model()
                self.spk_emb_func = get_embedding
            else:
                raise NotImplementedError

    def _logmelfilterbank(self, audio_for_mel):
        return logmelfilterbank(
            audio_for_mel,
            sampling_rate=self.config["sampling_rate"],
            hop_size=self.config["hop_size"],
            fft_size=self.config["fft_size"],
            win_length=self.config["win_length"],
            window=self.config["window"],
            num_mels=self.config["num_mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
            # keep compatibility
            log_base=self.config.get("log_base", 10.0),
        )

    def _extract_f0(self, audio_for_mel):
        if self.f0_extractor == "yaapt":
            f0 = get_yaapt_f0(
                audio_for_mel,
                rate=self.config["sampling_rate"],
                frame_length=self.config["fft_size"],
                frame_shift=self.config["hop_size"],
                interp=self.config["f0_interp"],
            )
        elif self.f0_extractor == "world":
            f0 = get_world_f0(
                audio_for_mel,
                fs=self.config["sampling_rate"],
                f0min=self.f0_min,
                f0max=self.f0_max,
                frame_length=self.config["fft_size"],
                frame_shift=self.config["hop_size"],
                interp=self.config["f0_interp"],
            )

        else:
            raise NotImplementedError

        if self.log_f0:
            lf0 = f0.copy()
            nonzero_indices = np.nonzero(f0)
            lf0[nonzero_indices] = np.log(f0[nonzero_indices])
            f0 = lf0
        if self.f0_normalize:
            # f0 = (f0 - self.f0_mean) / self.f0_scale
            f0 = f0 - self.f0_mean

        return f0

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray or tuple: Audio signal (T,) or (w/ sampling rate if return_sampling_rate = True).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio, fs = sf.read(self.audio_paths[utt_id])

        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        # audio /= 1 << (16 - 1)  # assume that wav is PCM 16 bit

        # resample the audio for logmelspec extraction if needed
        if fs != self.config["sampling_rate"]:
            audio_for_mel = librosa.resample(
                audio,
                orig_sr=fs,
                target_sr=self.config["sampling_rate"],
            )
        else:
            audio_for_mel = audio

        # extract logmelspec
        mel = self._logmelfilterbank(audio_for_mel)

        # always resample to 16kHz
        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)

        if self.return_sampling_rate:
            audio = (audio, fs)

        items = {"utt_id": "", "audio": audio, "mel": mel, "f0": None, "spemb": None, "wavpath": None}

        if self.return_utt_id:
            items["utt_id"] = utt_id
        if self.extract_f0:
            items["f0"] = self._extract_f0(audio_for_mel)
        if self.use_spk_emb:
            if self.spk_emb_source == "self":
                items["spemb"] = read_hdf5(self.spk_emb_paths[utt_id], "spemb")
            else:
                spembs = []
                for spk_emb_source_file in self.spk_emb_source_files[utt_id]:
                    spembs.append(
                        np.squeeze(
                            self.spk_emb_func(spk_emb_source_file, self.spk_emb_model)
                        )
                    )
                items["spemb"] = np.mean(np.stack(spembs, axis=0), axis=0)
        if self.return_wavpath:
            items["wavpath"] = self.audio_paths[utt_id]

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)


#######################
# Datasets with audio #
#######################


class AudioSCPMelDataset(MelDataset):
    """PyTorch compatible audio dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        config,
        wav_scp,
        spemb_scp=None,
        segments=None,
        *args,
        **kwargs,
    ):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            spemb_scp (str): Kaldi-style scp file for speaker embeddings
            segments (str): Kaldi-style segments file.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        super().__init__(config, **kwargs)

        audio_paths = dict()
        audio_keys = list()
        spk_emb_paths = dict()

        assert self.spk_emb_source in ["self", "external"], f"Unknown spk_emb_source: {self.spk_emb_source}"

        with open(wav_scp) as f:
            for line in f.read().splitlines():
                utt_id, contents = line.split(" ", 1)
                contents = contents.split(" ")
                audio_file = contents[0]
                audio_keys.append(utt_id)
                audio_paths[utt_id] = audio_file
                if self.use_spk_emb and self.spk_emb_source == "external":
                    assert (
                        len(contents[1:]) > 0
                    ), "during inference, please append speaker embedding source files at the end of each line."
                    spk_emb_paths[utt_id] = contents[1:]

        # during training, use pre-calculated speaker embeddings to accelerate
        if self.use_spk_emb and self.spk_emb_source == "self":
            assert spemb_scp is not None, "spemb_scp must be given during training"
            with open(spemb_scp) as f:
                for line in f.read().splitlines():
                    utt_id, h5path = line.split(" ")
                    spk_emb_paths[utt_id] = h5path

        self.audio_paths = audio_paths
        self.utt_ids = audio_keys
        self.spk_emb_paths = spk_emb_paths

        if self.allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]


class AudioMelDataset(MelDataset):
    """PyTorch compatible audio dataset based on a given directory of wav files."""

    def __init__(
        self,
        config,
        wavdir,
        query="*.wav",
        *args,
        **kwargs,
    ):
        """Initialize dataset.

        Args:
            wavdir (str): Kaldi-style wav.scp file.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        super().__init__(config, **kwargs)

        # find all of audio files
        audio_files = sorted(find_files(wavdir, query))
        audio_loader = {
            get_basename(audio_file): (sf.read(audio_file)[1], sf.read(audio_file)[0])
            for audio_file in audio_files
        }
        audio_keys = sorted(list(audio_loader.keys()))

        self.audio_loader = audio_loader
        self.utt_ids = audio_keys

        if self.allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

        # since wespeaker requires to extract from wav paths, save them in audio_files: {utt_id: path}
        self.audio_files = {
            get_basename(audio_file): audio_file for audio_file in audio_files
        }

        # audio paths is a must?
        self.audio_paths = self.audio_files


##########################
# Datasets without audio #
##########################


class FeatDataset(Dataset):
    """PyTorch compatible dataset given a directory of feature files."""

    def __init__(
        self,
        featdir,
        config,
        query,
        load_fn,
        return_utt_id=False,
        return_sampling_rate=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            wavdir (str): directory of wav files.
            segments (str): Kaldi-style segments file.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        self.config = config
        self.load_fn = load_fn

        # find files
        self.feat_files = sorted(find_files(featdir, query=query))
        self.utt_ids = [get_basename(path) for path in self.feat_files]

        self.return_utt_id = return_utt_id
        self.return_sampling_rate = return_sampling_rate
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray or tuple: Audio signal (T,) or (w/ sampling rate if return_sampling_rate = True).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        feats = self.load_fn(self.feat_files[idx])

        items = {
            "utt_id": "",
            "feat": feats,
        }

        if self.return_utt_id:
            items["utt_id"] = utt_id

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)
