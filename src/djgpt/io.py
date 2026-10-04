from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

SAMPLE_RATE = 22050   # 22k is plenty for analysis

def load_audio(path, sr=SAMPLE_RATE):
    """Load an audio file as mono at the target sample rate.
    
    Returns (waveform, sample_rate). Waveform is a 1-D float32 numpy array
    in [-1.0, 1.0]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, sample_rate = librosa.load(path, sr=sr, mono=True)
    return waveform, sample_rate

def save_audio(path, waveform, sr):
    """Write a mono waveform to disk as a WAV file. """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr)

def duration_seconds(waveform, sr):
    """Length of a waveform in seconds."""
    return len(waveform) / sr