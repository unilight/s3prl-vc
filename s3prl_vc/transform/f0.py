import numpy as np
import pyworld as pw
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

from s3prl_vc.utils.signal import low_cut_filter


def get_yaapt_f0(audio, rate=16000, frame_length=1024, frame_shift=256, interp=False):
    # convert frame_length and frame_shift from sample to ms
    frame_length_ms = int(frame_length / rate * 1000)
    frame_shift_ms = int(frame_shift / rate * 1000)

    # padding to match the shape of mel
    to_pad = frame_length // 2
    audio_pad = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)

    signal = basic.SignalObj(audio_pad, rate)
    pitch = pYAAPT.yaapt(
        signal,
        **{
            "frame_length": frame_length_ms,
            "frame_space": frame_shift_ms,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        }
    )
    if interp:
        return pitch.samp_interp
    else:
        return pitch.samp_values


def get_world_f0(
    x,
    fs,
    algorithm="dio",
    f0min=40,
    f0max=500,
    frame_length=1024,
    frame_shift=256,
    interp=False,
):
    frame_shift_ms = int(frame_shift / fs * 1000)
    x = x * np.iinfo(np.int16).max
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    if algorithm == "harvest":
        f = pw.harvest
    elif algorithm == "dio":
        f = pw.dio
    f0, _ = f(x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=frame_shift_ms)

    return f0
