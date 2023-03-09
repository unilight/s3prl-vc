# Template for any-to-one VC

This is a template recipe for training any-to-one VC models using your custom dataset. Several preparation steps are needed. Note when modifying them, keep in mind they were written w.r.t. the VCC2020 dataset. You might delete unnecessary or add more codes to your favor.

## Preparation

The following steps are NEEDED:

- Prepare your dataset and put it somewhere. There is no requirement on the directory structure, as long as the `local/data_prep.sh` can well generate the file lists used for training and decoding (conversion).
- Modify `conf/taco2_ar.yaml`: Modify fields like sampling rate, frame shift or custom vocoder to your preference.
- Modify `local/data_prep.sh`: this script needs to generate files containing space-separated lines with the format `<id> <wave file path>`, according to the directory structure of your custom dataset.

The following steps are OPTIONAL:

- Train your own vocoder. You can use the `hifigan_vctk+vcc2020` vocoder first, and see if you are satiffied with the quality. If not, please open an issue and I can guide you to train your own model. 
- `conf/f0.yaml` and `local/evaluate.py`: these files are for evaluation, which is optional depending on your application. Note that each evaluation metric has different requirements. For example, MCD, F0RMSE, F0CORR, DUR need parallel data. CER and WER need trnscription. If you have trouble modifying these files for your custom dataset, please open an issue and I will try to help you.

## Training

Run the following command:

```
./run.sh --stage -1 --stop_stage 2 --upstream <upstream> --trgspk <trgspk>
```

Four stages are executed:
- Stage -1: Pretrained model download. The `hifigan_vctk+vcc2020` will be downloaded (by default to `downloads/`).
- Stage 0: Data preparation. File lists should be generated in `data/` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training anc decoding (conversion)
- Stage 1: Statistics calculation. The statistics of the mel spectrogram used for normalzation is calculated using the training set of the target speaker. Calculation log and the statistics h5 file are saved in `data/` by default.
- Stage 2: Main training script. By default, `exp/<trgspk>_<upstream>_<taco2_ar>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--trgspk`: depending on your dataset, this can be conveniently used to train several A2O VC models.
- `--upstream`: In addition to the various upstreams provided by [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html), we also provide two PPG models: `ppg_sxliu` uses the ASR model provided by [Songxiang Liu's ppg-vc repo](https://github.com/liusongxiang/ppg-vc), and `ppg_whisper` uses the [OpenAI Whisper ASR model](https://github.com/openai/whisper). Note that in my experiments, the Whisper model yields very bad results, but I don't know what the reason is. I would appreciate it if someone could figure out why.
- `--tag`: if a tag is specified, results from stage 2 will be saved in `exp/<trgspk>_<upstream>_<tag>`.

## Decoding and evaluation
 
```
./run.sh --stage 3 --stop_stage 4 --upstream <upstream> --trgspk <trgspk> --checkpoint <checkpoint>
```

Generated files from both stages 3 and 4 are saved in `results/checkpoint-XXXXXsteps`.

- Stage 3 is the decoding stage and a log file is also generated. The mel spectrogram visualization can be viewed in `plot_mel/`. The generated waveform files are saved in `wav/`.
- Stage 4 is the evaluation stage using `local/evaluate.py`. MCD, F0RMSE, F0CORR, DUR, CER, WER are calculated. Detailed results are saved in `evaluation.log`.