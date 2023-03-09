from resemblyzer import VoiceEncoder, preprocess_wav


def load_asv_model():
    return VoiceEncoder(device="cpu")


def get_embedding(wav_path, model):
    wav = preprocess_wav(wav_path)
    return model.embed_utterance(wav)