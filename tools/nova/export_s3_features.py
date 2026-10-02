#!/usr/bin/env python3
"""Prepare a fixed 1000-frame speech fixture with installed S3 preprocessing."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    import s3tokenizer

    torch.set_num_threads(1)
    audio = s3tokenizer.load_audio(str(args.audio), sr=16000).cpu()
    if audio.ndim != 1 or audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    original_samples = audio.numel()
    # Keep model geometry identical for both backends. Extend shorter clips
    # with waveform silence, rather than inventing Mel padding values.
    audio = audio[:160000]
    if audio.numel() < 160000:
        audio = torch.nn.functional.pad(audio, (0, 160000 - audio.numel()))
    with torch.inference_mode():
        mel = s3tokenizer.log_mel_spectrogram(audio, n_mels=128)
    features = mel.detach().cpu().numpy().astype("<f4", copy=False)
    if features.shape != (128, 1000) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected feature shape/values: {features.shape}")
    # Refuse to overwrite an existing fixture.
    with args.output.open("xb") as stream:
        stream.write(features.tobytes(order="C"))
    print(f"audio={args.audio} original_samples={original_samples} "
          f"fixture_samples=160000 shape=1x128x1000 bytes={features.nbytes} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
