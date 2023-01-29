#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train VC model."""

import argparse
import logging
import os
import sys
import time

from collections import defaultdict, OrderedDict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from s3prl.nn import Featurizer

import s3prl_vc
import s3prl_vc.models
import s3prl_vc.losses

from s3prl_vc.upstream.interface import get_upstream
from s3prl_vc.datasets.datasets import AudioSCPMelDataset
from s3prl_vc.utils import read_hdf5
from s3prl_vc.utils.data import pad_list
from s3prl_vc.vocoder import Vocoder

# save for future use
# from s3prl_vc.utils.model_io import freeze_modules, filter_modules, get_partial_state_dict, transfer_verification, print_new_keys

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s3prl_vc.schedulers.schedulers import Linear_schedule_with_warmup

scheduler_classes = dict(linear_schedule_with_warmup=Linear_schedule_with_warmup)


class Trainer(object):
    """Customized trainer module for VC training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        upstream_model,
        upstream_featurizer,
        model,
        vocoder,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.upstream_model = upstream_model
        self.upstream_featurizer = upstream_featurizer
        self.model = model
        self.vocoder = vocoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
            state_dict["featurizer"] = self.upstream_featurizer.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()
            state_dict["featurizer"] = self.upstream_featurizer.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
            self.upstream_featurizer.module.load_state_dict(
                state_dict["upstream_featurizer"]
            )
        else:
            self.model.load_state_dict(state_dict["model"])
            self.upstream_featurizer.load_state_dict(state_dict["upstream_featurizer"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def load_trained_modules(self, checkpoint_path, init_mods):
        if self.config["distributed"]:
            main_state_dict = self.model.module.state_dict()
        else:
            main_state_dict = self.model.state_dict()

        if os.path.isfile(checkpoint_path):
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

            # first make sure that all modules in `init_mods` are in `checkpoint_path`
            modules = filter_modules(model_state_dict, init_mods)

            # then, actually get the partial state_dict
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)

            if partial_state_dict:
                if transfer_verification(main_state_dict, partial_state_dict, modules):
                    print_new_keys(partial_state_dict, modules, checkpoint_path)
                    main_state_dict.update(partial_state_dict)
        else:
            logging.error(f"Specified model was not found: {checkpoint_path}")
            exit(1)

        if self.config["distributed"]:
            self.model.module.load_state_dict(main_state_dict)
        else:
            self.model.load_state_dict(main_state_dict)

    def freeze_modules(self, freeze_mods):
        if self.config["distributed"]:
            freeze_modules(self.model.module, freeze_mods)
        else:
            freeze_modules(self.model, freeze_mods)

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        xs, ilens, ys, olens, spembs, f0s = tuple(
            [_.to(self.device) if _ is not None else _ for _ in batch]
        )

        # upstream forward
        with torch.no_grad():
            all_hs, all_hlens = self.upstream_model(xs, ilens)
        hs, hlens = self.upstream_featurizer(all_hs, all_hlens)

        # model forward
        outs, outs_lens = self.model(hs, hlens, targets=ys, spk_embs=spembs, f0s=f0s)

        # normalize output
        outs = (outs - self.config["trg_stats"]["mean"]) / self.config["trg_stats"][
            "scale"
        ]
        ys = (ys - self.config["trg_stats"]["mean"]) / self.config["trg_stats"]["scale"]

        # main loss
        gen_loss = self.criterion["main"](outs, outs_lens, ys, olens, self.device)
        self.total_train_loss["train/main"] += gen_loss.item()

        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.upstream_featurizer.parameters()),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""

        pass

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.upstream_featurizer.eval()
        self.model.eval()

        # save intermediate result
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # restore mode
        self.upstream_featurizer.train()
        self.model.train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(
            array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
        ):
            shape = array.shape
            if len(shape) == 1:
                # for eos probability
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(array)
                plt.xlabel("Frame")
                plt.ylabel("Probability")
                plt.ylim([0, 1])
            elif len(shape) == 2:
                # for tacotron 2 attention weights, whose shape is (out_length, in_length)
                if ref is None:
                    plt.figure(figsize=figsize, dpi=dpi)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                else:
                    plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                    plt.subplot(1, 2, 1)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                    plt.subplot(1, 2, 2)
                    plt.imshow(ref.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
            elif len(shape) == 4:
                # for transformer attention weights,
                # whose shape is (#leyers, #heads, out_length, in_length)
                plt.figure(
                    figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
                )
                for idx1, xs in enumerate(array):
                    for idx2, x in enumerate(xs, 1):
                        plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                        plt.imshow(x, aspect="auto")
                        plt.xlabel("Input")
                        plt.ylabel("Output")
            else:
                raise NotImplementedError("Support only from 1D to 4D array.")
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(figname)):
                # NOTE: exist_ok = True is needed for parallel process decoding
                os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname)
            plt.close()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # prepare input
        xs, ilens, ys, olens, spembs, f0s = tuple(
            [_.to(self.device) if _ is not None else _ for _ in batch]
        )
        if spembs is None:
            spembs = [None] * len(xs)
        if f0s is None:
            f0s = [None] * len(xs)

        # generate
        with torch.no_grad():
            all_hs, all_hlens = self.upstream_model(xs, ilens)
            hs, hlens = self.upstream_featurizer(all_hs, all_hlens)
            outs, _ = self.model(hs, hlens, spk_embs=spembs, f0s=f0s)

        for idx, (out, olen, y) in enumerate(zip(outs, olens, ys)):
            out = out[:olen]
            y = y[:olen]
            _plot_and_save(
                out.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y.cpu().numpy(),
                origin="lower",
            )

            if self.vocoder is not None:
                if not os.path.exists(os.path.join(dirname, "wav")):
                    os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)
                y, sr = self.vocoder.decode(out)
                sf.write(
                    os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                    y.cpu().numpy(),
                    sr,
                    "PCM_16",
                )

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self, use_f0=False):
        """Initialize customized collater for PyTorch DataLoader."""
        self.use_f0 = use_f0

    def __call__(self, batch):
        """Convert into batch tensors."""

        xs = [b["audio"] for b in batch]
        ys = [b["mel"] for b in batch]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)

        if self.use_f0:
            f0s = [b["f0"] for b in batch]
            f0s = pad_list([torch.from_numpy(f0).float() for f0 in f0s], 0)
        else:
            f0s = None

        return xs, ilens, ys, olens, None, f0s


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train VC model (See detail in bin/vc_train.py).")
    )
    parser.add_argument(
        "--upstream",
        required=True,
        type=str,
        help=("upstream model name. "),
    )
    parser.add_argument(
        "--train-scp",
        required=True,
        type=str,
        help=("directory including training wav scp. "),
    )
    parser.add_argument(
        "--dev-scp",
        required=True,
        type=str,
        help=("directory including source development data. "),
    )
    parser.add_argument(
        "--trg-stats",
        type=str,
        required=True,
        help="stats file for target denormalization.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help="yaml format configuration file (additional; for second-stage pretraining).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to initialize pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load main config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # load additional config
    if args.additional_config is not None:
        with open(args.additional_config) as f:
            additional_config = yaml.load(f, Loader=yaml.Loader)
        config.update(additional_config)

    # save config
    config["version"] = s3prl_vc.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # load target stats for denormalization
    config["trg_stats"] = {
        "mean": torch.from_numpy(read_hdf5(args.trg_stats, "mean")).float().to(device),
        "scale": torch.from_numpy(read_hdf5(args.trg_stats, "scale"))
        .float()
        .to(device),
    }

    # get dataset
    if config.get("use_f0", False):
        train_dataset = AudioSCPMelDataset(
            args.train_scp,
            config,
            extract_f0=config.get("use_f0", False),
            f0_extractor=config.get("f0_extractor", "world"),
            f0_min=read_hdf5(args.trg_stats, "f0_min"),  # for world f0 extraction
            f0_max=read_hdf5(args.trg_stats, "f0_max"),  # for world f0 extraction
            log_f0=config.get("log_f0", True),
            f0_normalize=config.get("f0_normalize", False),
            f0_mean=read_hdf5(args.trg_stats, "lf0_mean"),  # for speaker normalization
            f0_scale=read_hdf5(
                args.trg_stats, "lf0_scale"
            ),  # for speaker normalization
        )
        dev_dataset = AudioSCPMelDataset(
            args.dev_scp,
            config,
            extract_f0=config.get("use_f0", False),
            f0_extractor=config.get("f0_extractor", "world"),
            f0_min=read_hdf5(args.trg_stats, "f0_min"),  # for world f0 extraction
            f0_max=read_hdf5(args.trg_stats, "f0_max"),  # for world f0 extraction
            log_f0=config.get("log_f0", True),
            f0_normalize=config.get("f0_normalize", False),
            f0_mean=read_hdf5(args.trg_stats, "lf0_mean"),  # for speaker normalization
            f0_scale=read_hdf5(
                args.trg_stats, "lf0_scale"
            ),  # for speaker normalization
        )
    else:
        train_dataset = AudioSCPMelDataset(
            args.train_scp,
            config,
        )
        dev_dataset = AudioSCPMelDataset(
            args.dev_scp,
            config,
        )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(use_f0=config.get("use_f0", False))
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define upstream model
    upstream_model = get_upstream(args.upstream).to(device)
    upstream_model.eval()
    upstream_featurizer = Featurizer(upstream_model).to(device)

    # define models
    model_class = getattr(
        s3prl_vc.models,
        config.get("model_type", "Taco2_AR"),
    )
    model = model_class(
        upstream_featurizer.output_size,
        config["num_mels"],
        config["sampling_rate"]
        / config["hop_size"]
        * upstream_featurizer.downsample_rate
        / 16000,
        config["trg_stats"],
        **config["model_params"],
    ).to(device)

    # load vocoder
    if config.get("vocoder", False):
        vocoder = Vocoder(
            config["vocoder"]["checkpoint"],
            config["vocoder"]["config"],
            config["vocoder"]["stats"],
            config["trg_stats"],
            device,
        )
    else:
        vocoder = None

    # define criterions
    main_criterion_class = getattr(
        s3prl_vc.losses,
        config.get("main_loss_type", "L1"),
    )
    criterion = {
        "main": main_criterion_class(
            # keep compatibility
            **config.get("main_loss_params", {})
        ).to(device)
    }

    # define optimizers and schedulers
    optimizer_class = getattr(
        torch.optim,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        list(model.parameters()) + list(upstream_featurizer.parameters()),
        **config["optimizer_params"],
    )
    scheduler_class = scheduler_classes.get(
        config.get("scheduler_type", "linear_schedule_with_warmup")
    )
    scheduler = scheduler_class(
        optimizer=optimizer,
        num_training_steps=config["train_max_steps"],
        **config["scheduler_params"],
    )

    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        model = DistributedDataParallel(model)

    # show settings
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)
    logging.info(criterion)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        upstream_model=upstream_model,
        upstream_featurizer=upstream_featurizer,
        model=model,
        vocoder=vocoder,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.init_checkpoint) != 0:
        trainer.load_trained_modules(
            args.init_checkpoint, init_mods=config["init-mods"]
        )
        logging.info(f"Successfully load parameters from {args.init_checkpoint}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # freeze modules if necessary
    if config.get("freeze-mods", None) is not None:
        assert type(config["freeze-mods"]) is list
        trainer.freeze_modules(config["freeze-mods"])
        logging.info(f"Freeze modules with prefixes {config['freeze-mods']}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
