# Any-to-one VC recipe using the voice conversion challenge 2020 (VCC2020) dataset

Thie recipe can be used to reproduce the results in the [S3PRL-VC paper](https://arxiv.org/abs/2110.06280).

## Training

Run the following command:

```
./run.sh --stage -1 --stop_stage 2 --upstream <upstream> --trgspk <trgspk>
```

Four stages are executed:
- Stage -1: Data and pretrained model download. First, the VCC2020 dataset is downloaded (by default to `downloads/`). Then, several pre-trained vocoders are also downloaded. In the any-to-one VC setting, we use a Parallel WaveGAN vocoder.
- Stage 0: Data preparation. File lists are generated in `data/` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training anc decoding (conversion)
- Stage 1: Statistics calculation. The statistics of the mel spectrogram used for normalzation is calculated using the training set of the target speaker. Calculation log and the statistics h5 file are saved in `data/` by default.
- Stage 2: Main training script. By default, `exp/<trgspk>_<upstream>_<taco2_ar>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--trgspk`: The S3PRL-VC paper focused on task 1 in VCC2020. So, there are four target speakers to choose fromL TEF1, TEF2 (female), TEM1, TEM2 (male).
- `--upstream`: In addition to the various upstreams provided by [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html), we also provide two PPG models: `ppg_sxliu` uses the ASR model provided by [Songxiang Liu's ppg-vc repo](https://github.com/liusongxiang/ppg-vc), and `ppg_whisper` uses the [OpenAI Whisper ASR model](https://github.com/openai/whisper). Note that in my experiments, the Whisper model yields very bad results, but I don't know what the reason is. I would appreciate it if someone could figure out why.
- `--tag`: if a tag is specified, results from stage 2 will be saved in `exp/<trgspk>_<upstream>_<tag>`.

## Decoding (conversion) and evaluation

Run the following command:

```
./run.sh --stage 3 --stop_stage 4 --upstream <upstream> --trgspk <trgspk> --checkpoint <checkpoint>
```

Generated files from both stages 3 and 4 are saved in `results/checkpoint-XXXXXsteps`.

- Stage 3 is the decoding stage and a log file is also generated. The mel spectrogram visualization can be viewed in `plot_mel/`. The generated waveform files are saved in `wav/`.
- Stage 4 is the evaluation stage using `local/evaluate.py`. MCD, F0RMSE, F0CORR, DUR, CER, WER are calculated. Please refer to the code and the paper for what they represent. Detailed results are saved in `evaluation.log`.