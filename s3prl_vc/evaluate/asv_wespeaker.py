import numpy as np
import sys
import wespeakerruntime as wespeaker

def load_asv_model(device, lang="en"):
    return wespeaker.Inference(lang='chs')

def get_embedding(wav_path, model):
    return model.extract_embedding_wav(wav_path)

def get_cosine_similarity(x_emb, y_emb):
    return np.inner(x_emb, y_emb) / (np.linalg.norm(x_emb) * np.linalg.norm(y_emb))

def calculate_accept(x_path, y_path, model, threshold):
    x_emb = get_embedding(x_path, model)
    y_emb = get_embedding(y_path, model)
    cosine_similarity = get_cosine_similarity(x_emb, y_emb)
    return cosine_similarity > threshold
