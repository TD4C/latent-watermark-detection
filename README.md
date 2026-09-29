# AI Watermarking

**Latent-Space Embedding • Stable Diffusion • Verification • Robustness Evaluation**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-8A2BE2?style=for-the-badge)
![DDIM](https://img.shields.io/badge/Diffusion-DDIM-6C63FF?style=for-the-badge)
![Latent Watermark](https://img.shields.io/badge/Watermark-Latent%20Space%20%7C%20FFT-FF6F00?style=for-the-badge)
![Hamming](https://img.shields.io/badge/Error%20Correction-Hamming%287%2C4%29-00897B?style=for-the-badge)
![CRC32](https://img.shields.io/badge/Validation-CRC32-546E7A?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Interface-Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Evaluation](https://img.shields.io/badge/Evaluation-100%20Samples%20%7C%20300%20Robustness%20Cases-2E7D32?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed%20%7C%20Research%20Prototype-success?style=for-the-badge)


> [!NOTE]
> **Nội dung:** Đây là một research prototype triển khai watermark vô hình cho ảnh sinh bởi AI. Hệ thống kết hợp Stable Diffusion v1.5, MarianMT, watermark latent-space theo hướng Tree-Ring/FFT, Hamming(7,4), bit repetition và CRC32, đi kèm giao diện Gradio và các script đánh giá định lượng.

## 1. Tổng quan dự án

AI Watermark Project xây dựng một pipeline hoàn chỉnh để sinh ảnh AI có gắn watermark văn bản vô hình và xác minh watermark từ ảnh đầu ra.

### Pipeline chính

```text
Vietnamese Text / Prompt
          │
          ▼
       MarianMT
          │
          ▼
    English Prompt
          │
          ▼
 Stable Diffusion v1.5
          │
          │      Watermark Text
          │             │
          │             ▼
          │      MAGIC + LENGTH
          │      + PAYLOAD + CRC32
          │             │
          │             ▼
          │        Hamming(7,4)
          │             │
          │             ▼
          │       Bit Repetition ×3
          │             │
          ▼             │
      FFT / Ring Injection
          │
          ▼
   Watermarked Image
          │
          ▼
     DDIM Inversion
          │
          ▼
     FFT Correlation
          │
          ▼
     Hamming Decoding
          │
          ▼
   MAGIC + CRC Validation
          │
          ▼
    Extracted Text
```

Mục tiêu của project không chỉ là tạo watermark mà còn xây dựng một quy trình có thể:

- Tạo ảnh có watermark.
- Xác minh watermark.
- Đo detection/classification performance.
- Đo exact text extraction.
- Kiểm tra false positive trên ảnh sạch.
- Đánh giá robustness trước nhiều phép biến đổi ảnh.
- Lưu toàn bộ kết quả dưới dạng CSV để phân tích và tái sử dụng.

## 2. Mục tiêu

### 2.1. Watermarking

- Nhúng watermark văn bản vào latent representation thay vì hiển thị watermark trực tiếp trên ảnh.
- Sử dụng secret key để đồng bộ quá trình embedding và extraction.
- Hỗ trợ payload ngắn, có cấu trúc và kiểm tra toàn vẹn.

### 2.2. Error correction

- Sử dụng Hamming(7,4) để sửa lỗi bit.
- Lặp mỗi bit đã mã hóa 3 lần để tăng khả năng phục hồi.

### 2.3. Validation

- MAGIC xác nhận cấu trúc watermark.
- LENGTH xác định kích thước payload.
- CRC32 kiểm tra tính toàn vẹn sau giải mã.

### 2.4. Evaluation

Đánh giá trên **100 mẫu cân bằng**:

- 50 ảnh AI có watermark.
- 50 ảnh COCO sạch.

Tính confusion matrix và các classification metrics.

Đo detection rate, exact extraction rate và confidence.

### 2.5. Robustness

Đánh giá trên **50 ảnh positive** dưới 6 phép biến đổi:

1. JPEG Q30
2. Resize 256 → 512
3. Crop 80%
4. Gaussian Blur R2
5. Gaussian Noise
6. Rotation 5°

**Tổng cộng:** `50 samples × 6 attacks = 300 test cases`

## 3. Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Image Generation | Stable Diffusion v1.5 |
| Text Translation | Helsinki-NLP/opus-mt-vi-en |
| Diffusion Scheduler | DDIM |
| Image Inversion | DDIM Inverse Scheduler |
| Watermark Domain | Latent Space / FFT |
| Watermark Structure | MAGIC + LENGTH + PAYLOAD + CRC32 |
| Error Correction | Hamming(7,4) |
| Redundancy | Bit repetition ×3 |
| Interface | Gradio |
| Runtime | Python 3.10/3.11 |
| GPU | NVIDIA CUDA |
| Evaluation | NumPy / Pandas / CSV |
| Environment | VSCode / Windows |

## 4. Phương pháp watermark

### 4.1. Đóng gói payload

Watermark text được đóng gói theo cấu trúc:

```text
MAGIC + LENGTH + PAYLOAD + CRC32
```

Sau đó:

```text
bytes
  ↓
bits
  ↓
Hamming(7,4)
  ↓
repeat ×3
  ↓
FFT ring positions
  ↓
latent injection
```

### 4.2. Secret-keyed embedding

Secret được sử dụng để tạo seed xác định cho quá trình tạo vị trí watermark trong miền tần số.

Do đó:

```text
Same secret
    ↓
Same deterministic key / positions
    ↓
Embedding ↔ Extraction can be synchronized
```

> [!IMPORTANT]
> Secret phải giống nhau khi tạo và giải mã ảnh. Nếu ảnh được tạo bằng secret khác, quá trình extraction có thể không khôi phục được watermark.

## 5. Kiến trúc hệ thống

```text
┌───────────────────────────────────────────────────────┐
│                    Gradio Interface                   │
├───────────────────────────────┬───────────────────────┤
│         GENERATION            │      VERIFICATION     │
│                               │                       │
│ Vietnamese Prompt             │ Input Image           │
│        ↓                      │        ↓              │
│      MarianMT                 │   Resize 512×512      │
│        ↓                      │        ↓              │
│ Stable Diffusion              │   VAE Encode          │
│        ↓                      │        ↓              │
│ Watermark Injection           │   DDIM Inversion      │
│        ↓                      │        ↓              │
│ Watermarked Image             │ Watermark Extraction  │
│                               │        ↓              │
│                               │ MAGIC / CRC / Hamming │
└───────────────────────────────┴───────────────────────┘
```

## 6. Cấu trúc thư mục

```text
watermark_project/
│
├── watermark_system.py       # Core watermark + Stable Diffusion + Gradio
├── evaluate_100.py            # 50 positive + 50 negative + metrics
├── robustness_test.py         # 50 positive + 6 attacks + robustness CSV
├── requirements.txt           # Dependencies
├── README_VSCODE.md            # Documentation
│
├── data/
│   ├── coco/
│   │   ├── images/
│   │   └── annotations/
│   │
│   ├── positive_images/
│   │
│   └── robustness/
│       ├── original_images/
│       └── attack_images/
│
└── results/
    ├── watermark_evaluation_100_samples.csv
    ├── robustness_raw_results.csv
    └── robustness_summary.csv
```

> [!TIP]
> `data/` và các thư mục con cần thiết có thể được tạo tự động bởi script. Không cần tạo thủ công toàn bộ cấu trúc trước khi chạy.

## 7. Cài đặt môi trường

Khuyến nghị sử dụng **Python 3.10 hoặc 3.11** trên Windows.

### 7.1. Tạo virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 7.2. Cài PyTorch CUDA

Ví dụ môi trường đang sử dụng trong project:

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 7.3. Cài các dependency còn lại

```powershell
pip install -r requirements.txt
```

## 8. Kiểm tra môi trường

### 8.1. Kiểm tra PyTorch / CUDA

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

### 8.2. Kiểm tra NVIDIA driver

```powershell
nvidia-smi
```

Kỳ vọng trong môi trường GPU:

```text
Torch: 2.6.0+cu124
CUDA build: 12.4
CUDA available: True
GPU: NVIDIA GeForce ...
```

## 9. Cấu hình Secret

PowerShell:

```powershell
$env:WATERMARK_SECRET="CHANGE-ME-PRIVATE-WATERMARK-SECRET"
```

Nếu không đặt biến môi trường, code sử dụng giá trị mặc định hiện có trong chương trình.

> [!WARNING]
> Không chia sẻ secret thật của hệ thống lên GitHub nếu repository là public. Nên dùng biến môi trường khi triển khai thực tế.

## 10. Chạy hệ thống

### 10.1. Giao diện Gradio

```powershell
python watermark_system.py
```

Hệ thống cung cấp hai nhóm chức năng:

#### Generation

```text
Vietnamese Prompt
      ↓
  Translation
      ↓
Stable Diffusion
      ↓
Latent Watermark Injection
      ↓
Watermarked Image
```

#### Verification

```text
Input Image
      ↓
DDIM Inversion
      ↓
FFT-based Extraction
      ↓
Hamming Decoding
      ↓
MAGIC / CRC Validation
      ↓
Watermark Text + Confidence
```

## 11. Evaluation — 100 samples

Chạy lại benchmark bằng:

```powershell
python evaluate_100.py
```

### Thiết kế benchmark

| Group | Samples |
|---|---:|
| Positive — AI + watermark | 50 |
| Negative — clean COCO | 50 |
| **Total** | **100** |

Watermark định danh sử dụng chuỗi dạng:

```text
TD4C-0001
TD4C-0002
...
TD4C-0050
```

Ảnh negative được lấy từ COCO val2017 và được kỳ vọng không có watermark.

> [!NOTE]
> Project đã lưu sẵn kết quả benchmark vào `results/watermark_evaluation_100_samples.csv`. Vì vậy, có thể dùng trực tiếp CSV để xem và phân tích kết quả mà không cần chạy lại benchmark.

## 12. Kết quả Evaluation

### 12.1. Execution

| Metric | Result |
|---|---:|
| Total samples | 100 |
| Positive | 50 |
| Negative | 50 |
| Successful analysis | 100 |
| Processing errors | 0 |

### 12.2. Confusion Matrix

|  | Predicted Watermark | Predicted Clean |
|---|---:|---:|
| **Actual Watermark** | TP = 50 | FN = 0 |
| **Actual Clean** | FP = 0 | TN = 50 |

### 12.3. Classification Metrics

| Metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1-score | 100.00% |
| False Positive Rate | 0.00% |
| False Negative Rate | 0.00% |

### 12.4. Watermark Extraction

| Metric | Result |
|---|---:|
| Positive detection rate | 100.00% |
| Exact extraction rate | 100.00% |
| Average positive confidence | 0.6771 |

### 12.5. Clean COCO Test

| Metric | Result |
|---|---:|
| COCO images analyzed | 50 |
| False positive | 0 |
| COCO FPR | 0.00% |

## 13. Robustness Evaluation

Chạy:

```powershell
python robustness_test.py
```

6 attack được áp dụng độc lập trên cùng 50 ảnh positive:

1. `JPEG_Q30`
2. `RESIZE_256`
3. `CROP_80`
4. `BLUR_R2`
5. `GAUSSIAN_NOISE`
6. `ROTATE_5DEG`

**Tổng:** `50 × 6 = 300 test cases`

Kết quả chi tiết:

```text
results/robustness_raw_results.csv
```

Kết quả tổng hợp:

```text
results/robustness_summary.csv
```

## 14. Robustness Results

| Attack | Samples | Detection | Exact Extraction | FNR | Avg. Confidence | Avg. Corrected Bits |
|---|---:|---:|---:|---:|---:|---:|
| JPEG Q30 | 50 | 94.00% | 94.00% | 6.00% | 0.5272 | 1.66 |
| Resize 256→512 | 50 | 100.00% | 100.00% | 0.00% | 0.6304 | 0.22 |
| Crop 80% | 50 | 0.00% | 0.00% | 100.00% | 0.1565 | 8.84 |
| Gaussian Blur R2 | 50 | 98.00% | 98.00% | 2.00% | 0.5747 | 0.56 |
| Gaussian Noise | 50 | 98.00% | 98.00% | 2.00% | 0.5659 | 1.14 |
| Rotation 5° | 50 | 0.00% | 0.00% | 100.00% | 0.1753 | 8.70 |

### 14.1. Execution status

```text
Total test cases       : 300
Successful analyses    : 300
Processing errors      : 0
```

### 14.2. Observed behavior

```text
Resize 256→512     → 100% detection
Blur R2             → 98% detection
Gaussian Noise      → 98% detection
JPEG Q30            → 94% detection
Crop 80%            → 0% detection
Rotation 5°         → 0% detection
```

Các kết quả cho thấy watermark hiện tại duy trì khả năng trích xuất tốt hơn trước các biến đổi mức điểm ảnh/chất lượng như resize, blur, noise và JPEG ở cấu hình thử nghiệm. Ngược lại, crop và rotation làm suy giảm mạnh khả năng đồng bộ cần thiết cho quá trình inversion/extraction.

> [!WARNING]
> Các tỷ lệ trên là kết quả thực nghiệm của đúng cấu hình attack đã kiểm thử, không nên diễn giải thành độ bền tuyệt đối trước mọi mức nén, crop, blur, noise hoặc rotation.

## 15. Phân tích confidence và corrected_bits

Ngoài `detected_valid`, project lưu:

- `confidence`: mức tương quan/độ tin cậy của extraction.
- `corrected_bits`: số bit được Hamming decoder sửa.
- `reason`: lý do validation thành công/thất bại.

Một xu hướng quan sát được trong benchmark là:

```text
Watermark ổn định
      ↓
confidence cao hơn
      ↓
ít corrected_bits hơn
```

Trong các trường hợp watermark suy giảm mạnh, confidence giảm và corrected_bits tăng.

### Ví dụ trong robustness benchmark

#### CROP_80

```text
confidence    ≈ 0.1565
corrected_bits ≈ 8.84
detection      = 0%
```

#### ROTATE_5DEG

```text
confidence    ≈ 0.1753
corrected_bits ≈ 8.70
detection      = 0%
```

Các thống kê này hữu ích cho việc phân tích lỗi và tìm ngưỡng robustness trong các thí nghiệm tiếp theo.

## 16. Các file CSV

### `watermark_evaluation_100_samples.csv`

Một record tương ứng với một mẫu trong benchmark 100 ảnh.

Các trường quan trọng gồm:

```text
sample_id
sample_type
expected_watermark
detected_valid
extracted_text
watermark_exact_match
confidence
corrected_bits
classification_correct
success
reason
```

Dùng để:

- Kiểm tra từng mẫu.
- Tái tính TP/TN/FP/FN.
- Đối chiếu watermark expected/extracted.
- Phân tích confidence.
- Kiểm tra lỗi extraction.

### `robustness_summary.csv`

Một record cho mỗi attack.

Dùng trực tiếp cho bảng tổng hợp trong báo cáo:

```text
attack
total_samples
successful_analysis
errors
detected_watermark
detection_rate_percent
exact_extraction_rate_percent
false_negative_rate_percent
avg_confidence
avg_corrected_bits
```

### `robustness_raw_results.csv`

Chứa dữ liệu chi tiết của:

```text
50 samples × 6 attacks = 300 records
```

Dùng cho:

- Phân tích từng sample.
- Tìm các trường hợp fail.
- Kiểm tra extracted text.
- Theo dõi confidence.
- Theo dõi corrected bits.
- Đối chiếu nguyên nhân failure.

## 17. Reproducibility

Để so sánh các lần chạy một cách nhất quán, nên giữ nguyên:

- Model / checkpoint
- Secret
- Watermark configuration
- Seed
- Scheduler
- Inference steps
- Input dataset
- Attack parameters

Các yếu tố có thể làm kết quả thay đổi:

- Model version
- PyTorch / CUDA environment
- Random seed
- Scheduler
- Secret
- Số inference steps
- Tập ảnh
- Tham số attack

> [!TIP]
> Nếu đã có 3 CSV trong `results/`, không cần chạy lại benchmark chỉ để xem kết quả cũ. Hai script `evaluate_100.py` và `robustness_test.py` được giữ lại để phục vụ tái lập thí nghiệm khi cần.

## 18. Hạn chế hiện tại

### Dataset

Bộ benchmark hiện tại gồm 100 mẫu, trong đó 50 positive và 50 negative. Quy mô này phù hợp với một evaluation cơ bản nhưng chưa đại diện cho toàn bộ phân phối ảnh AI và ảnh tự nhiên.

### Positive set

Positive images được sinh bởi chính pipeline watermarking của project và được xác minh bằng cơ chế extraction tương ứng. Do đó, đây chủ yếu là một self-consistency benchmark cho pipeline hiện tại.

### Geometric robustness

Trong cấu hình thử nghiệm hiện tại:

```text
Crop 80%    → 0%
Rotation 5° → 0%
```

Đây là hạn chế nổi bật của cấu hình watermark/extraction hiện tại.

### Robustness scope

Benchmark mới kiểm thử một mức cường độ cho mỗi attack. Không nên suy rộng trực tiếp các kết quả này sang mọi mức attack.

## 19. Hướng phát triển

Các hướng mở rộng:

- Benchmark nhiều mức JPEG.
- Benchmark nhiều mức crop.
- Benchmark nhiều góc rotation.
- Nhiều mức blur và noise.
- Tăng số lượng ảnh.
- Bổ sung các nguồn ảnh AI độc lập.
- Đánh giá cross-model.
- Nghiên cứu geometric synchronization/alignment.
- Tối ưu latent watermark để tăng robustness.
- Bổ sung biểu đồ và báo cáo tự động từ CSV.

Một protocol mở rộng có thể sử dụng các mức như:

| Attack | Các mức đề xuất |
|---|---|
| JPEG | 90 / 70 / 50 / 30 / 10 |
| Blur | 1 / 2 / 3 / 5 |
| Noise | 5 / 10 / 15 / 20 / 30 |
| Crop | 5 / 10 / 15 / 20 / 30% |
| Rotation | 1 / 2 / 3 / 5 / 10° |

## 20. Cách chạy nhanh

### Chỉ dùng GUI

```powershell
.\.venv\Scripts\activate
python watermark_system.py
```

### Chạy lại evaluation

```powershell
.\.venv\Scripts\activate
python evaluate_100.py
```

### Chạy lại robustness

```powershell
.\.venv\Scripts\activate
python robustness_test.py
```

> [!WARNING]
> Không nên chạy đồng thời nhiều script có cùng model diffusion nếu GPU có VRAM hạn chế.

## 21. Repository workflow

```text
watermark_system.py
        │
        ├── Generate
        │      ↓
        │  Watermarked Images
        │
        └── Verify
               ↓
          Extraction / Validation

evaluate_100.py
        ↓
100-sample benchmark
        ↓
watermark_evaluation_100_samples.csv

robustness_test.py
        ↓
6 attacks × 50 samples
        ↓
robustness_raw_results.csv
robustness_summary.csv
```

## 22. License

License của repository nên được xác định và đặt trong file `LICENSE` ở thư mục gốc trước khi public repository.

Hiện README không tự xác nhận một license cụ thể nếu repository chưa có file license chính thức.

## 23. Project Status

**STATUS: COMPLETED — Research Prototype**

Project hiện bao gồm:

-  AI image generation
-  Invisible latent watermark embedding
-  Hamming error correction
-  CRC validation
-  Watermark extraction
-  Gradio interface
-  100-sample evaluation
-  300-case robustness evaluation
-  CSV-based experiment results
-  VSCode / Windows project structure

## 24. Quick Reference

| Task | Command |
|---|---|
| Activate environment | `.\.venv\Scripts\activate` |
| Run GUI | `python watermark_system.py` |
| Run 100-sample evaluation | `python evaluate_100.py` |
| Run robustness benchmark | `python robustness_test.py` |
| Check CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| Check NVIDIA driver | `nvidia-smi` |

---

> [!NOTE]
> **Research note:** Các số liệu trong README được lấy từ benchmark hiện tại của project và được giữ nguyên theo các file CSV kết quả đã tạo. Khi thay đổi model, dataset, secret, seed hoặc attack parameters, cần thực hiện lại benchmark trước khi so sánh.
