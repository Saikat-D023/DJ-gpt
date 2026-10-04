"""Phase 1 smoke test — load, inspect, round-trip two samples."""

from src.djgpt.io import load_audio, save_audio, duration_seconds

SAMPLES = ["samples/sample1.mp3", "samples/sample2.mp3"]

for i, path in enumerate(SAMPLES, start=1):
    waveform, sr = load_audio(path)
    print(
        f"Song {i}: {path}\n"
        f"  shape      = {waveform.shape}\n"
        f"  sample rate= {sr} Hz\n"
        f"  duration   = {duration_seconds(waveform, sr):.1f}s\n"
        f"  peak       = {abs(waveform).max():.3f}\n"
        f"  rms        = {(waveform ** 2).mean() ** 0.5:.3f}"
    )
    save_audio(f"outputs/roundtrip_{i}.wav", waveform, sr)
    print(f"  wrote      = outputs/roundtrip_{i}.wav\n")

print("Phase 1 done. Play the roundtrip WAVs in any audio player.")