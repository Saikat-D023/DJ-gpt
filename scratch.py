from src.djgpt.io import load_audio, duration_seconds
from src.djgpt.analysis import (
    detect_tempo, detect_key, segment, energy_curve, segment_energy,
)

SAMPLES = ["samples/sample1.mp3", "samples/sample2.mp3", "samples/sample3.mp3"]

for i, path in enumerate(SAMPLES, start=1):
    print(f"\n=== Song {i}: {path} ===")
    waveform, sr = load_audio(path)
    print(f"duration: {duration_seconds(waveform, sr):.1f}s")

    bpm, beat_times = detect_tempo(waveform, sr)
    print(f"tempo:    {bpm:.1f} BPM ({len(beat_times)} beats detected)")

    root, mode = detect_key(waveform, sr)
    print(f"key:      {root} {mode}")

    segments = segment(waveform, sr, n_segments=8)
    e_times, e_values = energy_curve(waveform, sr)

    print(f"segments ({len(segments)}):")
    for s in segments:
        avg_e = segment_energy(s, e_times, e_values)
        bar = "#" * int(avg_e * 100)
        print(
            f"  [{s.index}] {s.start_sec:6.1f}s - {s.end_sec:6.1f}s "
            f"({s.duration_sec:5.1f}s)  energy={avg_e:.3f} {bar}"
        )

print("\nPhase 2 done.")