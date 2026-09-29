"""Evaluate the watermark system on 50 positive + 50 clean COCO images.

Run:
    python evaluate_100.py

The script creates/reuses 50 AI images with embedded text watermarks,
then analyzes them and 50 COCO val2017 images, finally writing a CSV
and printing TP/TN/FP/FN + Accuracy/Precision/Recall/F1/FPR/FNR.
"""

from __future__ import annotations

import gc
import json
import random
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm

from watermark_system import AIGCWatermarkEngine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COCO_DIR = DATA_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "images"
COCO_ANNOTATIONS_DIR = COCO_DIR / "annotations"
POSITIVE_DIR = DATA_DIR / "positive_images"
RESULTS_DIR = PROJECT_DIR / "results"

COCO_ANNOTATION_FILE = (
    COCO_ANNOTATIONS_DIR / "instances_val2017.json"
)
COCO_ANNOTATION_ZIP = (
    COCO_DIR / "annotations_trainval2017.zip"
)

POSITIVE_CSV = RESULTS_DIR / "positive_50_results.csv"
COCO_CSV = RESULTS_DIR / "coco_50_negative_results.csv"
FINAL_CSV = RESULTS_DIR / "watermark_evaluation_100_samples.csv"

POSITIVE_COUNT = 50
NEGATIVE_COUNT = 50
SEED = 2026
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 8.0

COCO_ANNOTATION_URL = (
    "http://images.cocodataset.org/annotations/"
    "annotations_trainval2017.zip"
)
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/"

for folder in (
    COCO_IMAGES_DIR,
    COCO_ANNOTATIONS_DIR,
    POSITIVE_DIR,
    RESULTS_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)


PROMPTS = [
    "một chú mèo cam nằm trên ghế sofa trong phòng khách ấm áp",
    "một chú chó golden retriever chạy trên bãi cỏ xanh",
    "một chiếc ô tô thể thao đỗ trên con đường thành phố hiện đại",
    "một chiếc xe đạp dựng cạnh quán cà phê ngoài trời",
    "một con chim đầy màu sắc đậu trên cành cây",
    "một con hươu đứng giữa khu rừng xanh vào buổi sáng",
    "một con voi đi qua đồng cỏ rộng lớn dưới bầu trời xanh",
    "một con ngựa chạy trên bãi biển lúc hoàng hôn",
    "một con gấu trúc ngồi ăn tre trong khu rừng",
    "một chú thỏ trắng ngồi giữa vườn hoa",
    "một thành phố hiện đại với những tòa nhà cao tầng",
    "một con phố đông người vào buổi tối",
    "một cây cầu lớn bắc qua dòng sông",
    "một ngôi nhà nhỏ bên hồ giữa thiên nhiên",
    "một căn phòng hiện đại với cửa sổ lớn",
    "một bàn ăn gia đình với nhiều món ăn",
    "một chiếc bánh sinh nhật nhiều tầng trên bàn",
    "một tách cà phê bên cạnh quyển sách",
    "một chiếc điện thoại thông minh đặt trên bàn làm việc",
    "một chiếc máy tính xách tay trong văn phòng hiện đại",
    "một chiếc đồng hồ cổ đặt trên bàn gỗ",
    "một bó hoa hồng đặt trong chiếc bình thủy tinh",
    "một khu vườn đầy hoa dưới ánh nắng",
    "một cánh đồng lúa xanh trải dài đến chân trời",
    "một ngọn núi cao với tuyết phủ trên đỉnh",
    "một hồ nước trong xanh giữa những ngọn núi",
    "một bãi biển nhiệt đới với hàng dừa",
    "một chiếc thuyền buồm trên đại dương",
    "một ngọn hải đăng bên bờ biển",
    "một thác nước lớn giữa khu rừng",
    "một lâu đài cổ nằm trên ngọn đồi",
    "một ngôi chùa cổ giữa phong cảnh thiên nhiên",
    "một khu chợ ngoài trời đông đúc",
    "một cửa hàng bánh ngọt hiện đại",
    "một nhà hàng sang trọng với ánh đèn vàng",
    "một con đường làng yên bình giữa cánh đồng",
    "một căn nhà gỗ trong khu rừng",
    "một cabin nhỏ bên hồ nước",
    "một chiếc máy bay bay trên bầu trời",
    "một đoàn tàu chạy qua vùng nông thôn",
    "một chiếc xe máy chạy trên con đường ven biển",
    "một người đang chụp ảnh phong cảnh thiên nhiên",
    "một chiếc máy ảnh chuyên nghiệp trên bàn",
    "một khu cắm trại giữa rừng thông",
    "một lều cắm trại bên cạnh dòng suối",
    "một con thuyền nhỏ trên mặt hồ yên tĩnh",
    "một khu rừng mùa thu với lá vàng",
    "một con phố thành thị hiện đại vào buổi sáng",
    "một con phố cổ với những ngôi nhà đầy màu sắc",
    "một quảng trường thành phố với đài phun nước",
]

if len(PROMPTS) != POSITIVE_COUNT:
    raise ValueError(
        f"Cần {POSITIVE_COUNT} prompt, hiện có {len(PROMPTS)}."
    )


# ============================================================
# HELPERS
# ============================================================


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def ensure_engine() -> AIGCWatermarkEngine:
    print("[MODEL] Khởi tạo AIGCWatermarkEngine...")
    return AIGCWatermarkEngine()


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def ensure_coco_annotations() -> None:
    if COCO_ANNOTATION_FILE.exists():
        print(f"[COCO] Annotation đã tồn tại: {COCO_ANNOTATION_FILE}")
        return

    print("[COCO] Đang tải annotations_trainval2017.zip...")
    download_file(COCO_ANNOTATION_URL, COCO_ANNOTATION_ZIP)

    print("[COCO] Đang giải nén annotation...")
    with zipfile.ZipFile(COCO_ANNOTATION_ZIP, "r") as zf:
        zf.extractall(COCO_DIR)

    if not COCO_ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {COCO_ANNOTATION_FILE} sau khi giải nén."
        )


def load_coco_metadata() -> tuple[list[dict], dict[int, set[str]]]:
    ensure_coco_annotations()

    with COCO_ANNOTATION_FILE.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_map = {
        c["id"]: c["name"] for c in coco["categories"]
    }

    image_categories: dict[int, set[str]] = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        image_categories.setdefault(image_id, set())
        if category_id in category_map:
            image_categories[image_id].add(category_map[category_id])

    return coco["images"], image_categories


def prepare_positive_samples(engine: AIGCWatermarkEngine) -> pd.DataFrame:
    """Create or reuse the 50 positive samples and save metadata."""
    existing = None
    if POSITIVE_CSV.exists():
        try:
            existing = pd.read_csv(POSITIVE_CSV)
        except Exception:
            existing = None

    rows: list[dict] = []

    for i, prompt in enumerate(tqdm(PROMPTS, desc="Positive images"), start=1):
        watermark = f"TD4C-{i:04d}"
        image_path = POSITIVE_DIR / f"positive_{i:03d}.png"

        try:
            if image_path.exists():
                translated_prompt = ""
            else:
                torch.manual_seed(SEED + i - 1)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + i - 1)

                image, translated_prompt = engine.generate(
                    prompt=prompt,
                    watermark_text=watermark,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                )
                image.convert("RGB").save(image_path, format="PNG")
                del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            rows.append(
                {
                    "sample_id": i,
                    "expected_watermark": watermark,
                    "prompt": prompt,
                    "translated_prompt": translated_prompt,
                    "image_path": str(image_path.resolve()),
                    "generation_success": True,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": i,
                    "expected_watermark": watermark,
                    "prompt": prompt,
                    "translated_prompt": "",
                    "image_path": str(image_path.resolve()),
                    "generation_success": False,
                    "generation_error": str(exc),
                }
            )
            print(f"[ERROR] positive {i}: {exc}")
            traceback.print_exc()

    positive_df = pd.DataFrame(rows)
    positive_df.to_csv(POSITIVE_CSV, index=False, encoding="utf-8-sig")
    return positive_df


def analyze_positive(
    engine: AIGCWatermarkEngine,
    positive_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    print("\n[1/2] Phân tích 50 ảnh AI có watermark...")

    for _, row in tqdm(positive_df.iterrows(), total=len(positive_df), desc="Analyze positive"):
        image_path = Path(row["image_path"])
        expected = str(row["expected_watermark"])

        base = {
            "sample_type": "positive",
            "expected_valid": True,
            "expected_watermark": expected,
            "prompt": row["prompt"],
            "translated_prompt": row.get("translated_prompt", ""),
            "file_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "image_id": "",
            "category": "",
        }

        try:
            image = Image.open(image_path).convert("RGB")
            result = engine.invert_and_extract(
                image,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

            detected = bool(result.get("valid", False))
            extracted = result.get("text", "")
            exact = detected and extracted == expected

            rows.append(
                {
                    **base,
                    "detected_valid": detected,
                    "extracted_text": extracted,
                    "watermark_exact_match": exact,
                    "confidence": float(result.get("confidence", 0.0)),
                    "corrected_bits": int(result.get("corrected", 0)),
                    "classification_correct": detected,
                    "success": True,
                    "reason": result.get("reason", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "detected_valid": None,
                    "extracted_text": "",
                    "watermark_exact_match": False,
                    "confidence": None,
                    "corrected_bits": None,
                    "classification_correct": False,
                    "success": False,
                    "reason": f"ANALYSIS_ERROR: {exc}",
                }
            )
            print(f"[ERROR] positive {image_path.name}: {exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def download_and_analyze_coco(
    engine: AIGCWatermarkEngine,
) -> pd.DataFrame:
    images, image_categories = load_coco_metadata()
    rng = random.Random(SEED)
    selected = rng.sample(images, NEGATIVE_COUNT)

    downloaded: list[tuple[dict, Path]] = []
    print(f"\n[COCO] Chọn {NEGATIVE_COUNT} ảnh val2017...")

    for info in tqdm(selected, desc="Download COCO"):
        file_name = info["file_name"]
        image_path = COCO_IMAGES_DIR / file_name
        if not image_path.exists():
            download_file(COCO_IMAGE_URL + file_name, image_path)
        downloaded.append((info, image_path))

    rows: list[dict] = []
    print("\n[2/2] Phân tích 50 ảnh COCO sạch...")

    for info, image_path in tqdm(downloaded, desc="Analyze COCO"):
        categories = ", ".join(sorted(image_categories.get(info["id"], set())))
        base = {
            "sample_type": "negative",
            "expected_valid": False,
            "expected_watermark": "",
            "prompt": "",
            "translated_prompt": "",
            "file_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "image_id": info["id"],
            "category": categories,
        }

        try:
            image = Image.open(image_path).convert("RGB")
            result = engine.invert_and_extract(
                image,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

            detected = bool(result.get("valid", False))
            rows.append(
                {
                    **base,
                    "detected_valid": detected,
                    "extracted_text": result.get("text", ""),
                    "watermark_exact_match": False,
                    "confidence": float(result.get("confidence", 0.0)),
                    "corrected_bits": int(result.get("corrected", 0)),
                    "classification_correct": not detected,
                    "success": True,
                    "reason": result.get("reason", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "detected_valid": None,
                    "extracted_text": "",
                    "watermark_exact_match": False,
                    "confidence": None,
                    "corrected_bits": None,
                    "classification_correct": False,
                    "success": False,
                    "reason": f"ANALYSIS_ERROR: {exc}",
                }
            )
            print(f"[ERROR] COCO {image_path.name}: {exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def compute_metrics(final_df: pd.DataFrame) -> dict:
    eval_df = final_df[final_df["success"] == True].copy()

    tp = int(((eval_df["expected_valid"] == True) & (eval_df["detected_valid"] == True)).sum())
    tn = int(((eval_df["expected_valid"] == False) & (eval_df["detected_valid"] == False)).sum())
    fp = int(((eval_df["expected_valid"] == False) & (eval_df["detected_valid"] == True)).sum())
    fn = int(((eval_df["expected_valid"] == True) & (eval_df["detected_valid"] == False)).sum())

    total = len(eval_df)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    pos = eval_df[eval_df["expected_valid"] == True]
    pos_detection = pos["detected_valid"].mean() if len(pos) else 0.0
    exact_rate = pos["watermark_exact_match"].mean() if len(pos) else 0.0
    avg_conf = pos["confidence"].mean() if len(pos) else 0.0

    neg = eval_df[eval_df["expected_valid"] == False]
    neg_fpr = neg["detected_valid"].mean() if len(neg) else 0.0

    return {
        "total": len(final_df),
        "success": total,
        "errors": len(final_df) - total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "PositiveDetectionRate": pos_detection,
        "ExactExtractionRate": exact_rate,
        "AvgPositiveConfidence": avg_conf,
        "COCOFPR": neg_fpr,
    }


def main() -> None:
    print("=" * 72)
    print("WATERMARK EVALUATION - 50 POSITIVE + 50 COCO")
    print("=" * 72)

    print(f"Project: {PROJECT_DIR}")
    print(f"Results: {RESULTS_DIR}")

    engine = ensure_engine()

    positive_meta = prepare_positive_samples(engine)
    positive_results = analyze_positive(engine, positive_meta)
    positive_results.to_csv(POSITIVE_CSV, index=False, encoding="utf-8-sig")

    negative_results = download_and_analyze_coco(engine)
    negative_results.to_csv(COCO_CSV, index=False, encoding="utf-8-sig")

    required_columns = [
        "sample_type",
        "expected_valid",
        "expected_watermark",
        "detected_valid",
        "extracted_text",
        "watermark_exact_match",
        "confidence",
        "corrected_bits",
        "classification_correct",
        "success",
        "reason",
        "prompt",
        "translated_prompt",
        "file_name",
        "image_path",
        "image_id",
        "category",
    ]

    for df in (positive_results, negative_results):
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

    positive_final = positive_results[required_columns].copy()
    negative_final = negative_results[required_columns].copy()

    for df in (positive_final, negative_final):
        for col in ("expected_valid", "detected_valid", "success", "watermark_exact_match"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: None if pd.isna(x) and col == "detected_valid" else normalize_bool(x)
                )

    final_df = pd.concat([positive_final, negative_final], ignore_index=True)
    if len(final_df) != POSITIVE_COUNT + NEGATIVE_COUNT:
        raise RuntimeError(f"Dataset cuối có {len(final_df)} mẫu, không phải 100.")

    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    final_df.insert(0, "sample_id", range(1, len(final_df) + 1))
    final_df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    metrics = compute_metrics(final_df)

    print("\n" + "=" * 72)
    print("KẾT QUẢ")
    print("=" * 72)
    print(f"Tổng số mẫu               : {metrics['total']}")
    print(f"Phân tích thành công      : {metrics['success']}")
    print(f"Lỗi xử lý                 : {metrics['errors']}")
    print("\n--- CONFUSION MATRIX ---")
    print(f"TP = {metrics['TP']}")
    print(f"TN = {metrics['TN']}")
    print(f"FP = {metrics['FP']}")
    print(f"FN = {metrics['FN']}")
    print("\n--- CLASSIFICATION METRICS ---")
    print(f"Accuracy                 : {metrics['Accuracy'] * 100:.2f}%")
    print(f"Precision                : {metrics['Precision'] * 100:.2f}%")
    print(f"Recall                   : {metrics['Recall'] * 100:.2f}%")
    print(f"F1-score                 : {metrics['F1'] * 100:.2f}%")
    print(f"False Positive Rate      : {metrics['FPR'] * 100:.2f}%")
    print(f"False Negative Rate      : {metrics['FNR'] * 100:.2f}%")
    print("\n--- WATERMARK EXTRACTION ---")
    print(f"Positive detection rate  : {metrics['PositiveDetectionRate'] * 100:.2f}%")
    print(f"Exact extraction rate    : {metrics['ExactExtractionRate'] * 100:.2f}%")
    print(f"Avg positive confidence  : {metrics['AvgPositiveConfidence']:.4f}")
    print("\n--- COCO NEGATIVE TEST ---")
    print(f"COCO FPR                 : {metrics['COCOFPR'] * 100:.2f}%")
    print("\n--- FILES ---")
    print(f"Positive CSV             : {POSITIVE_CSV}")
    print(f"COCO CSV                 : {COCO_CSV}")
    print(f"Final CSV                : {FINAL_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
