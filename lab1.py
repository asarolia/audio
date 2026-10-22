"""Lab 1: audio fundamentals you can see and hear.

    uv run lab1.py frames              # bytes per 20 ms frame at each stage
    uv run lab1.py record              # 5 s from the mic, per-frame stats, saves out/mic_48k.wav
    uv run lab1.py alias               # chirp sweep: naive vs correct downsampling
    uv run lab1.py phone [--input F]   # browser audio -> simulated PSTN path -> back to 16 kHz
    uv run lab1.py all

Everything is written to ./out/.
"""

import argparse
import queue
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

import g711

OUT = Path(__file__).parent / "out"
FRAME_MS = 20


def samples_per_frame(sample_rate: int, frame_ms: int = FRAME_MS) -> int:
    return sample_rate * frame_ms // 1000


def dbfs(frame: np.ndarray) -> float:
    """RMS level relative to full scale. 0 dBFS is the loudest an int16 can be."""
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return 20 * np.log10(max(rms, 1e-9) / 32768)


def save_spectrogram(x: np.ndarray, sr: int, title: str, path: Path) -> None:
    plt.figure(figsize=(9, 3.5))
    plt.specgram(x, Fs=sr, NFFT=512, noverlap=384, cmap="magma")
    plt.ylim(0, 24000 if sr >= 48000 else sr / 2)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("Hz")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# --- Step 0: frame arithmetic -------------------------------------------------

def cmd_frames(_args) -> None:
    stages = [
        ("Browser capture, int16", 48000, 2),
        ("STT input, int16", 16000, 2),
        ("PSTN on the wire, G.711", 8000, 1),
        ("PSTN decoded, int16", 8000, 2),
    ]
    print(f"{'stage':28} {'rate':>6} {'samples/20ms':>13} {'bytes/20ms':>11} {'kbps':>7}")
    for name, sr, bytes_per_sample in stages:
        n = samples_per_frame(sr)
        print(f"{name:28} {sr:>6} {n:>13} {n * bytes_per_sample:>11} {sr * bytes_per_sample * 8 / 1000:>7.0f}")
    print("\nSame 20 ms of audio, very different byte counts. Size buffers in samples or ms, not bytes.")


# --- Step 1: capture ----------------------------------------------------------

def cmd_record(args) -> None:
    import sounddevice as sd

    dev = sd.query_devices(kind="input")
    print(f"Input device: {dev['name']}  (native rate {dev['default_samplerate']:.0f} Hz)")

    sr = 48000
    frames: queue.Queue = queue.Queue()

    def callback(indata, n_frames, time_info, status):
        # This runs on the real-time audio thread. Copy the data and return
        # straight away: printing, file I/O or locks here cause dropouts.
        if status:
            frames.put(("status", str(status)))
        frames.put(("audio", indata[:, 0].copy()))

    captured = []
    print(f"Recording {args.seconds}s... say: fifty, sixty, ship, sip, five thousand, nine thousand, Shreya, Sriya")
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                        blocksize=samples_per_frame(sr), callback=callback):
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            kind, payload = frames.get()
            if kind == "status":
                print(f"  !! {payload}")   # overflow means we lost audio
                continue
            captured.append(payload)
            i = len(captured)
            if i % 10 == 1:    # print every 10th frame (every 200 ms) to keep it readable
                print(f"  frame {i:4d}: {len(payload)} samples, {payload.dtype}, "
                      f"{payload.nbytes} bytes, min {payload.min():6d}, max {payload.max():6d}, "
                      f"{dbfs(payload):6.1f} dBFS")

    audio = np.concatenate(captured)
    OUT.mkdir(exist_ok=True)
    sf.write(OUT / "mic_48k.wav", audio, sr, subtype="PCM_16")
    print(f"Saved {len(captured)} frames ({len(audio) / sr:.2f}s) to out/mic_48k.wav")


# --- Step 2: aliasing ---------------------------------------------------------

def cmd_alias(_args) -> None:
    OUT.mkdir(exist_ok=True)
    sr = 48000
    t = np.arange(int(sr * 4)) / sr
    sweep = 0.5 * signal.chirp(t, f0=0, f1=20000, t1=t[-1], method="linear")

    naive = sweep[::3]                         # just drop 2 of every 3 samples
    proper = signal.resample_poly(sweep, 1, 3)  # low-pass below 8 kHz first, then decimate

    save_spectrogram(sweep, sr, "Original sweep, 48 kHz (0 to 20 kHz)", OUT / "alias_0_original.png")
    save_spectrogram(naive, 16000, "Naive x[::3] to 16 kHz", OUT / "alias_1_naive.png")
    save_spectrogram(proper, 16000, "resample_poly to 16 kHz", OUT / "alias_2_proper.png")
    for name, x, rate in [("original_48k", sweep, sr), ("naive_16k", naive, 16000), ("proper_16k", proper, 16000)]:
        sf.write(OUT / f"alias_{name}.wav", x.astype(np.float32), rate)
    print("Wrote out/alias_*.png and .wav. Predict the naive plot before you open it.")


# --- Step 3 and 4: simulated phone path ----------------------------------------

def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def cmd_phone(args) -> None:
    OUT.mkdir(exist_ok=True)
    src = Path(args.input) if args.input else OUT / "mic_48k.wav"
    audio, sr = sf.read(src, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 48000:
        raise SystemExit(f"Expected 48 kHz input, got {sr} Hz")

    # What a good web pipeline gives the STT model.
    wide_16k = to_int16(signal.resample_poly(audio, 1, 3))

    # What the PSTN gives you: 8 kHz, telephone band (~300-3400 Hz), 8-bit A-law.
    narrow_8k = signal.resample_poly(audio, 1, 6)
    band = signal.butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")
    narrow_8k = to_int16(signal.sosfilt(band, narrow_8k))

    alaw_bytes = g711.linear_to_alaw(narrow_8k)        # this is what travels in RTP
    decoded_8k = g711.alaw_to_linear(alaw_bytes)
    wrong_law_8k = g711.ulaw_to_linear(alaw_bytes)      # the codec-mismatch bug

    # Upsample back to 16 kHz for an STT model trained on wideband audio.
    phone_16k = to_int16(signal.resample_poly(decoded_8k, 2, 1))

    n = samples_per_frame(8000)
    print(f"20 ms frame: {n} A-law bytes on the wire -> {decoded_8k[:n].nbytes} bytes after decoding")
    err = decoded_8k.astype(np.float64) - narrow_8k
    snr = 10 * np.log10(np.mean(narrow_8k.astype(np.float64) ** 2) / max(np.mean(err ** 2), 1e-9))
    print(f"A-law quantisation SNR: {snr:.1f} dB")

    outputs = [
        ("phone_0_wideband_16k", wide_16k, 16000),
        ("phone_1_alaw_decoded_8k", decoded_8k, 8000),
        ("phone_2_wrong_law_8k", wrong_law_8k, 8000),
        ("phone_3_upsampled_16k", phone_16k, 16000),
    ]
    for name, x, rate in outputs:
        sf.write(OUT / f"{name}.wav", x, rate, subtype="PCM_16")
    (OUT / "phone_alaw.raw").write_bytes(alaw_bytes.tobytes())

    save_spectrogram(wide_16k, 16000, "Wideband 16 kHz (what the web gives you)", OUT / "phone_spec_wideband.png")
    save_spectrogram(phone_16k, 16000, "Phone path upsampled to 16 kHz", OUT / "phone_spec_upsampled.png")
    print("Wrote out/phone_*.wav, phone_alaw.raw and phone_spec_*.png. Listen on headphones.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frames")
    r = sub.add_parser("record")
    r.add_argument("--seconds", type=float, default=5)
    sub.add_parser("alias")
    ph = sub.add_parser("phone")
    ph.add_argument("--input", help="48 kHz WAV (default: out/mic_48k.wav)")
    a = sub.add_parser("all")
    a.add_argument("--seconds", type=float, default=5)
    a.add_argument("--input", default=None)
    args = p.parse_args()

    if args.cmd == "all":
        cmd_frames(args)
        cmd_record(args)
        cmd_alias(args)
        cmd_phone(args)
    else:
        {"frames": cmd_frames, "record": cmd_record, "alias": cmd_alias, "phone": cmd_phone}[args.cmd](args)


if __name__ == "__main__":
    main()
