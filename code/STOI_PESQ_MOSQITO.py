# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:03:49 2025

@author: Stefan
"""

import numpy as np
import soundfile as sf

# Install these with:
# pip install pystoi pesq mosqito
#import PIL
import matplotlib as mpl
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    data_path = mpl.get_data_path()
except AttributeError:
    # Newer matplotlib versions
    data_path = os.path.join(mpl.get_configdir(), 'mpl-data')
    
from pystoi import stoi
import torchaudio
import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
import torchmetrics.functional.audio as audio_metrics

import mosqito
from collections.abc import Iterable

def load_audio(file_path, target_fs):
    data, fs = sf.read(file_path)
    if fs != target_fs:
        import librosa
        data = librosa.resample(data, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, fs

def flatten_all(data):
    flat_list = []
    def _flatten(x):
        if isinstance(x, (np.ndarray, list, tuple)):
            # For NumPy arrays, also check if it's scalar or shape=()
            if isinstance(x, np.ndarray) and x.shape == ():
                flat_list.append(x.item())
            else:
                for elem in x:
                    _flatten(elem)
        elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            # Handle any other iterable except strings, bytes
            for elem in x:
                _flatten(elem)
        else:
            # Scalar or unhandled type, append directly
            flat_list.append(x) 
    _flatten(data)
    return np.array(flat_list)

def chunkwise_mosqito_metrics(ref_audio, deg_audio, fs=48000, chunk_duration_sec=10):
    chunk_size = chunk_duration_sec * fs
    num_chunks = int(np.ceil(len(ref_audio) / chunk_size))

    loudness_ref_scores = []
    sharpness_ref_scores = []
    roughness_ref_scores = []

    loudness_deg_scores = []
    sharpness_deg_scores = []
    roughness_deg_scores = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size

        ref_chunk = ref_audio[start:end]
        deg_chunk = deg_audio[start:end]

        # Skip if chunk is too short to calculate meaningful metrics
        if len(ref_chunk) < fs:  # at least 1 second chunk
            continue

        # Calculate metrics for reference chunk
        loudness_ref_scores.append(mosqito.loudness_zwtv(ref_chunk, fs=fs))
        sharpness_ref_scores.append(mosqito.sharpness_din_tv(ref_chunk, fs=fs))
        roughness_ref_scores.append(mosqito.roughness_dw(ref_chunk, fs=fs))

        # Calculate metrics for degraded chunk
        loudness_deg_scores.append(mosqito.loudness_zwtv(deg_chunk, fs=fs))
        sharpness_deg_scores.append(mosqito.sharpness_din_tv(deg_chunk, fs=fs))
        roughness_deg_scores.append(mosqito.roughness_dw(deg_chunk, fs=fs))

    # Prepare for averaging if array is ragged
    loudness_ref_scores=flatten_all(loudness_ref_scores)
    sharpness_ref_scores=flatten_all(sharpness_ref_scores)
    roughness_ref_scores=flatten_all(roughness_ref_scores)
        
    loudness_deg_scores=flatten_all(loudness_deg_scores)
    sharpness_deg_scores=flatten_all(sharpness_deg_scores)
    roughness_deg_scores=flatten_all(roughness_deg_scores)
        
    # Average the collected scores for each metric
    mosqito_scores = {
    "loudness_ref":  np.mean(loudness_ref_scores) if loudness_ref_scores.any() else float('nan'),
    "sharpness_ref": np.mean(sharpness_ref_scores) if sharpness_ref_scores.any() else float('nan'),
    "roughness_ref": np.mean(roughness_ref_scores) if roughness_ref_scores.any() else float('nan'),
    "loudness_deg": np.mean(loudness_deg_scores) if loudness_deg_scores.any() else float('nan'),
    "sharpness_deg": np.mean(sharpness_deg_scores) if sharpness_deg_scores.any() else float('nan'),
    "roughness_deg": np.mean(roughness_deg_scores) if roughness_deg_scores.any() else float('nan'),
    }

    return mosqito_scores

def compute_metrics(ref_path, deg_path):
    # Load reference and degraded audio at 16kHz mono
    ref16k, fs16k = load_audio(ref_path, 16000)
    deg16k, fs_deg = load_audio(deg_path, 16000)
    assert fs16k == fs_deg, "Sampling rates must match."
    
    ref48k, fs48k = load_audio(ref_path, 48000)
    deg48k, fs_deg = load_audio(deg_path, 48000)
    assert fs48k == fs_deg, "Sampling rates must match."

    # Make sure arrays are same length
    min_len16k = min(len(ref16k), len(deg16k))
    ref16k = ref16k[:min_len16k]
    deg16k = deg16k[:min_len16k]
    
    min_len48k = min(len(ref48k), len(deg48k))
    ref48k = ref48k[:min_len48k]
    deg48k = deg48k[:min_len48k]
    
    # STOI (Short-Time Objective Intelligibility)
    stoi_score = stoi(ref16k, deg16k, fs16k, extended=False)

    # PESQ (Perceptual Evaluation of Speech Quality)
    # mode='wb' means wide-band, use sample rate 16000
    #pesq_score = pesq(fs, ref, deg, 'wb')
    # Load your audio as PyTorch tensors with shape (channels, samples)
    ref_tensor = torch.tensor(np.array(ref16k))  # make sure it’s a tensor, not ndarray
    deg_tensor = torch.tensor(np.array(deg16k))

    # Both tensors should be floats and on CPU
    ref_tensor = ref_tensor.float()
    deg_tensor = deg_tensor.float()
    
    def chunk_audio(audio_tensor, chunk_size_samples):
        for start in range(0, audio_tensor.shape[-1], chunk_size_samples):
            yield audio_tensor[..., start:start + chunk_size_samples]

    # Parameters
    chunk_duration_sec = 10
    chunk_size = chunk_duration_sec * fs16k

    # Assuming ref_tensor and deg_tensor are loaded audio tensors (channels, samples)
    pesq_scores = []

    for ref_chunk, deg_chunk in zip(chunk_audio(ref_tensor, chunk_size), chunk_audio(deg_tensor, chunk_size)):
        # Check chunks are long enough (PESQ requires minimum length)
        if ref_chunk.shape[-1] < 160:  # minimal length check (adjust accordingly)
            continue
        score = audio_metrics.perceptual_evaluation_speech_quality(deg_chunk, ref_chunk, fs=fs16k, mode="wb")
        pesq_scores.append(score.item())

    overall_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else float('nan')
    print(f"Overall PESQ score (averaged over chunks): {overall_pesq:.4f}")
    
    # MOSQITO (Speech Quality) - uses sample rate 48000Hz internally
    # Calculate MoSQITo metrics
    avg_mosqito_metrics = chunkwise_mosqito_metrics(ref48k, deg48k)

    return stoi_score, overall_pesq, avg_mosqito_metrics


if __name__ == "__main__":
    #ref_file = "D:\PRIVAT\Musik Projekte\python\Digital Convolution Audio Filter by Frequency Bin Specific Decayed Discrete Impulse Response\speech.wav"
    #deg_file = "D:\PRIVAT\Musik Projekte\python\Digital Convolution Audio Filter by Frequency Bin Specific Decayed Discrete Impulse Response\speech_filtered.wav"
    ref_file = "D:\PRIVAT\Musik Projekte\python\Digital Convolution Audio Filter by Frequency Bin Specific Decayed Discrete Impulse Response\Record.wav"
    deg_file = "D:\PRIVAT\Musik Projekte\python\Digital Convolution Audio Filter by Frequency Bin Specific Decayed Discrete Impulse Response\Record_filtered_bareIR.wav"

    stoi_score, overall_pesq, avg_mosqito_metrics = compute_metrics(ref_file, deg_file)
    print(f"STOI: {stoi_score:.4f}")
    print(f"PESQ: {overall_pesq:.4f}")
    print("mosqito metrics:", avg_mosqito_metrics)
