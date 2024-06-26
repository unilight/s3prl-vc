import wespeakerruntime as wespeaker


def load_asv_model(lang="en"):
    return wespeaker.Inference(lang=lang)


def get_embedding(wav_path, model):
    return model.extract_embedding_wav(wav_path)
