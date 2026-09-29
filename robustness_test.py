"""Robustness test for the watermark system.

Run:
    python robustness_test.py

Requires the 50 positive samples from evaluate_100.py. If they do not
exist yet, this script can create them automatically through the shared
helper in evaluate_100.py.
"""

from __future__ import annotations

import gc
import io
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from watermark_system import AIGCWatermarkEngine
from evaluate_100 import (
    POSITIVE_COUNT,
    POSITIVE_CSV,
    POSITIVE_DIR,
    PROMPTS,
    RESULTS_DIR,
    SEED,
    NUM_INFERENCE_STEPS,
    prepare_positive_samples,
)

PROJECT_DIR = Path(__file__).resolve().parent
ROBUSTNESS_DIR = DATA_DIR = PROJECT_DIR / "data" / "robustness"
ORIGINAL_DIR = ROBUSTNESS_DIR / "original_images"
ATTACK_DIR = ROBUSTNESS_DIR / "attack_images"
RAW_CSV = RESULTS_DIR / "robustness_raw_results.csv"
SUMMARY_CSV = RESULTS_DIR / "robustness_summary.csv"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
ATTACK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def attack_jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def attack_resize(image: Image.Image) -> Image.Image:
    result = image.resize((256, 256), Image.Resampling.LANCZOS)
    result = result.resize((512, 512), Image.Resampling.LANCZOS)
    return result.convert("RGB")

def attack_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    crop_w = int(w * 0.80)
    crop_h = int(h * 0.80)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    result = image.crop((left, top, left + crop_w, top + crop_h))
    result = result.resize((512, 512), Image.Resampling.LANCZOS)
    return result.convert("RGB")

def attack_blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=2.0)).convert("RGB")


def attack_noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(SEED).normal(0.0, 12.0, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).convert("RGB")


def attack_rotate(image: Image.Image) -> Image.Image:
    return image.rotate(
        5,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(128, 128, 128),
    ).convert("RGB")


ATTACKS = {
    "JPEG_Q30": attack_jpeg,
    "RESIZE_256": attack_resize,
    "CROP_80": attack_crop,
    "BLUR_R2": attack_blur,
    "GAUSSIAN_NOISE": attack_noise,
    "ROTATE_5DEG": attack_rotate,
}

def load_or_create_positive(engine: AIGCWatermarkEngine) -> pd.DataFrame:
    if POSITIVE_CSV.exists() and all(
        (POSITIVE_DIR / f"positive_{i:03d}.png").exists()
        for i in range(1, POSITIVE_COUNT + 1)
    ):
        df = pd.read_csv(POSITIVE_CSV)
        if len(df) >= POSITIVE_COUNT:
            return df.iloc[:POSITIVE_COUNT].copy()

    print("[POSITIVE] Chưa đủ 50 ảnh, đang tạo lại...")
    return prepare_positive_samples(engine)


def main() -> None:
    print("=" * 80)
    print("ROBUSTNESS TEST - 50 ẢNH × 6 ATTACK")
    print("=" * 80)

    engine = AIGCWatermarkEngine()
    positive_df = load_or_create_positive(engine)

    samples = []
    for i, row in positive_df.iloc[:POSITIVE_COUNT].iterrows():
        samples.append(
            {
                "sample_id": int(row.get("sample_id", i + 1)),
                "watermark": str(row["expected_watermark"]),
                "prompt": str(row.get("prompt", "")),
                "image_path": str(row["image_path"]),
            }
        )

    results: list[dict] = []

    for attack_name, attack_fn in ATTACKS.items():
        print("\n" + "-" * 70)
        print(f"ATTACK: {attack_name}")
        print("-" * 70)

        attack_folder = ATTACK_DIR / attack_name
        attack_folder.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=attack_name):
            sample_id = sample["sample_id"]
            original_path = Path(sample["image_path"])
            attacked_path = attack_folder / f"sample_{sample_id:03d}.png"
            expected = sample["watermark"]

            try:
                image = Image.open(original_path).convert("RGB")
                attacked_image = attack_fn(image)
                attacked_image.save(attacked_path, format="PNG")

                result = engine.invert_and_extract(
                    attacked_image,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                )

                detected = bool(result.get("valid", False))
                extracted = result.get("text", "")

                results.append(
                    {
                        "sample_id": sample_id,
                        "attack": attack_name,
                        "expected_valid": True,
                        "expected_watermark": expected,
                        "detected_valid": detected,
                        "extracted_text": extracted,
                        "exact_match": detected and extracted == expected,
                        "detection_correct": detected,
                        "confidence": float(result.get("confidence", 0.0)),
                        "corrected_bits": int(result.get("corrected", 0)),
                        "success": True,
                        "reason": result.get("reason", ""),
                        "original_path": str(original_path.resolve()),
                        "attacked_path": str(attacked_path.resolve()),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "sample_id": sample_id,
                        "attack": attack_name,
                        "expected_valid": True,
                        "expected_watermark": expected,
                        "detected_valid": None,
                        "extracted_text": "",
                        "exact_match": False,
                        "detection_correct": False,
                        "confidence": None,
                        "corrected_bits": None,
                        "success": False,
                        "reason": f"ERROR: {exc}",
                        "original_path": str(original_path.resolve()),
                        "attacked_path": str(attacked_path.resolve()),
                    }
                )
                print(f"[ERROR] {attack_name} sample={sample_id}: {exc}")
                traceback.print_exc()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(
                    RAW_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    summary_rows = []
    for attack_name in ATTACKS:
        attack_df = raw_df[raw_df["attack"] == attack_name].copy()
        valid_df = attack_df[attack_df["success"] == True].copy()

        successful = len(valid_df)
        detected = int(valid_df["detected_valid"].sum()) if successful else 0
        exact = int(valid_df["exact_match"].sum()) if successful else 0
        detection_rate = detected / successful if successful else 0.0
        exact_rate = exact / successful if successful else 0.0
        fnr = 1.0 - detection_rate if successful else 0.0
        avg_conf = valid_df["confidence"].mean() if successful else 0.0
        avg_corrected = valid_df["corrected_bits"].mean() if successful else 0.0

        summary_rows.append(
            {
                "attack": attack_name,
                "total_samples": len(attack_df),
                "successful_analysis": successful,
                "errors": len(attack_df) - successful,
                "detected_watermark": detected,
                "detection_rate": detection_rate,
                "detection_rate_percent": detection_rate * 100,
                "exact_extraction": exact,
                "exact_extraction_rate": exact_rate,
                "exact_extraction_rate_percent": exact_rate * 100,
                "false_negative_rate": fnr,
                "false_negative_rate_percent": fnr * 100,
                "avg_confidence": avg_conf,
                "avg_corrected_bits": avg_corrected,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("KẾT QUẢ ROBUSTNESS TEST")
    print("=" * 90)
    print(
        summary_df[
            [
                "attack",
                "total_samples",
                "successful_analysis",
                "errors",
                "detected_watermark",
                "detection_rate_percent",
                "exact_extraction_rate_percent",
                "false_negative_rate_percent",
                "avg_confidence",
                "avg_corrected_bits",
            ]
        ].to_string(index=False)
    )

    print("\nRAW CSV:")
    print(RAW_CSV)
    print("\nSUMMARY CSV:")
    print(SUMMARY_CSV)
    print("=" * 90)


if __name__ == "__main__":
    main()
