""" Pull musical info from the waveform, no ML yet."""

from dataclasses import dataclass
from turtle import mode
import numpy as np
import librosa

# Krumhansl-Schmuckler key profiles — correlation templates for the 12 pitch
# classes. One for major, one for minor. These are standard values from the
# music cognition literature; don't tweak them.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

@dataclass
class Segment:
    start_sec: float
    end_sec: float
    index: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

def detect_tempo(waveform, sr):
    """ EStimate BPM and Timestamps of each beat. """
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
    bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

    if bpm < 70:
        bpm *= 2
    elif bpm > 180:
        bpm /= 2

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return bpm, beat_times

def detect_key(waveform, sr):
    """ Estimate song key as root and mode (major/minor) using Krumhansl-Schmuckler key profiles. """

    # Average chroma across the whole song
    chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # shape: (12,)

    # Correlate against all 24 key profiles (12 roots x 2 modes)
    best_score = -np.inf
    best_root = "C"
    best_mode = "major"

    for i in range(12):
        major_rotated = np.roll(_MAJOR_PROFILE, i)
        minor_rotated = np.roll(_MINOR_PROFILE, i)
        major_score = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_score = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
        if major_score > best_score:
            best_score, best_root, best_mode = major_score, _PITCH_CLASSES[i], "major"
        if minor_score > best_score:
            best_score, best_root, best_mode = minor_score, _PITCH_CLASSES[i], "minor"

    return best_root, best_mode

def segment(waveform, sr, n_segments: int = 8):
    """Split the song into n_segments structural chunks.

    Uses agglomerative clustering on a chroma self-similarity matrix.
    Returns a list of Segments with start/end times in seconds.
    """
    chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
    boundary_frames = librosa.segment.agglomerative(chroma, k=n_segments)
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)

    # Append the end of the song as the final boundary
    duration = len(waveform) / sr
    boundary_times = np.append(boundary_times, duration)

    return [
        Segment(start_sec=float(boundary_times[i]),
                end_sec=float(boundary_times[i + 1]),
                index=i)
        for i in range(len(boundary_times) - 1)
    ]

def energy_curve(waveform, sr, hop_sec: float = 0.5):
    """RMS energy over short windows
    
    Returns (times, rms_values) where both are 1-D arrays and times[i] is the
    center of the window that produced rms_values[i].
    """
    hop_length = int(hop_sec * sr)
    frame_length = hop_length * 2
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms

def segment_energy(segment: Segment, energy_times: np.ndarray, energy_values: np.ndarray) -> float:
    """Average RMS energy inside one segment. Useful for ranking segments."""
    mask = (energy_times >= segment.start_sec) & (energy_times < segment.end_sec)
    if not mask.any():
        return 0.0
    return float(energy_values[mask].mean())