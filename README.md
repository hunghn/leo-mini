# LEO-MINI

Tái hiện thực nghiệm bài báo **"LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts"** (EMNLP 2025).

> Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki — University of Waterloo

Repo này có **hai variant**:

| Variant | LLM | Vision experts | N^V | Benchmark | Mục đích |
|---------|-----|----------------|-----|-----------|----------|
| **LEO-MINI** (gốc) | Llama-3.2-1B/3B, Phi-3.5-mini | CLIP + EVA-02 + ConvNeXt + Pix2Struct | 64 | MME, POPE, TextVQA, … | Tái hiện paper |
| **Vi-LEO-MINI** | Qwen2.5-3B-Instruct | CLIP + Pix2Struct | 128 | **ViTextVQA** | Hỏi đáp ảnh có chữ tiếng Việt |

---

## Tổng quan

LEO-MINI giải quyết hai vấn đề cốt lõi của MLLM hiện đại:

**Hiệu quả (Efficiency):** Visual tokens chiếm phần lớn context length của LLM.  
LEO-MINI giảm 576 tokens/expert xuống còn **64 tokens** bằng module **CoTR** (Conditional Token Reduction).

**Hiệu năng (Effectiveness):** Nhiều domain yêu cầu features đặc thù.  
LEO-MINI dùng **MMoE** (Mixture of Multi-Modal Experts) — nhiều vision expert đa dạng và LoRA expert có routing thông minh trong LLM.

| So sánh | LEO-MINI-Llama3-8B | LLaVA-NeXT (2880 tokens) |
|---------|---------------------|--------------------------|
| Visual tokens | **64** | 2880 |
| FLOPs | **−63.83%** | baseline |
| CUDA time | **−90.69%** | baseline |
| MME | **1583** | 1533 |
| POPE | **90.3** | 86.5 |
| TextVQA | **75.1** | 67.6 |

---

## Kiến trúc

### LEO-MINI (paper gốc — 4 experts)

```
Image(s) + Text Instruction
       │
       ├─ MMoE-Vision (4 experts — tất cả frozen trong Stage 3)
       │     ├─ CLIP ViT-L/14-336       → (B, 576, 1024)
       │     ├─ EVA-02 CLIP-L-14-336    → (B, 576, 1024)
       │     ├─ ConvNeXt-Large-D        → (B, 576,  768)
       │     └─ Pix2Struct-Large        → (B, 576, 2048)  d^V tổng = 4864
       │
       ├─ CoTR  [Stage 3 only]  576×4 → 64 tokens/expert
       │     Eq.2  s_QUERY  = Q̄_i · Ī_i^T          ∈ R^{N^V × N_i}
       │     Eq.3  s_SELF   = 1·Ī_i · Ī_i^T         ∈ R^{1 × N_i}
       │     Eq.4  s_CROSS  = Σ_{j≠i} 1·Ī_j · Ī_i^T ∈ R^{1 × N_i}
       │     Eq.5  s_TEXT   = 1·T̂ · Ī_i^T           ∈ R^{1 × N_i}
       │     Eq.6  α_i = softmax((Σ scores) / √d_i^V)
       │     Eq.7  Ī_i = α_i · I_i  →  concat → (B, 64, 4864)
       │
       ├─ Visual Projector  (B, 64, 4864) → (B, 64, d_LLM)
       │     Linear → GELU → Linear
       │
       └─ LLM + MMoE-LLM [Stage 3 only]
             concat([Ī_proj, T_embed]) → LLM
             down_proj replaced by:
               y = f_ORI(x) + f_GEN(x) + f_top1(x)        (Eq.8)
             Router: 2-layer MLP + GELU ← (Ī, T, x)
             Balance loss: λ·Σ_i(fraction_i − 1/E)²       (λ=0.05)
```

### Vi-LEO-MINI (Vietnamese TextVQA — 2 experts)

```
Image (tiếng Việt có chữ) + Câu hỏi tiếng Việt
       │
       ├─ MMoE-Vision (2 experts — frozen trong Stage 3)
       │     ├─ CLIP ViT-L/14-336       → (B, 576, 1024)  nhận biết cảnh tổng quát
       │     └─ Pix2Struct-Large        → (B, 576, 2048)  chuyên OCR / document
       │                                  d^V tổng = 3072
       │
       ├─ CoTR  [Stage 3 only]  576×2 → 128 tokens   (N^V=128 để giữ detail chữ)
       │
       ├─ Visual Projector  (B, 128, 3072) → (B, 128, 2048)
       │
       └─ Qwen2.5-3B-Instruct + MMoE-LLM [Stage 3 only]
             Chat template: <|im_start|>user / <|im_end|> / <|im_start|>assistant
```

**Lý do chọn cấu hình này:**
- **Bỏ EVA-02 + ConvNeXt**: hai expert này tối ưu cho object detection / scene understanding, ít giá trị cho ViTextVQA (scene-text heavy). Tiết kiệm ~1.8 GB VRAM.
- **Tăng N^V từ 64 → 128**: ảnh có chữ tiếng Việt (biển hiệu, hóa đơn, biển số) cần nhiều spatial tokens hơn để giữ nguyên vẹn hình dạng ký tự.
- **Qwen2.5-3B**: hỗ trợ tiếng Việt tốt hơn Llama-3.2 nhờ dữ liệu pretraining đa ngôn ngữ phong phú hơn.

### 3-Stage Training (Table 6, paper)

| Stage | Trainable | Frozen | Data |
|-------|-----------|--------|------|
| 1 — Warmup Projector | Visual Projector | LLM, Vision Experts | EAGLE alignment / **KTVIC** (captioning) |
| 2 — Full SFT | Tất cả | — | EAGLE SFT / **OpenViVQA** (general VQA) |
| 3 — Token Reduction | **CoTR + MMoE-LLM + Projector** | LLM backbone, Vision Experts | LLaVA-v1.5 665K / **ViTextVQA** (scene-text) |

---

## Base Models được hỗ trợ

| Model | VRAM Stage 2 | VRAM Stage 3 | Optimizer Stage 2 | Ghi chú |
|-------|-------------|-------------|-------------------|---------|
| `meta-llama/Llama-3.2-1B-Instruct` | ~14 GB | ~8 GB | adamw_torch | LEO-MINI gốc |
| `meta-llama/Llama-3.2-3B-Instruct` | ~27 GB | ~14 GB | **adamw_bnb_8bit** | LEO-MINI gốc |
| `microsoft/Phi-3.5-mini-instruct` | ~30 GB | ~16 GB | **adamw_bnb_8bit** | LEO-MINI gốc |
| `Qwen/Qwen2.5-3B-Instruct` | ~27 GB | ~14 GB | **adamw_bnb_8bit** | **Vi-LEO-MINI** |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ~65 GB | ~32 GB | DeepSpeed Zero2 | Paper gốc (8× GPU) |

> **Llama-3.2-3B**, **Phi-3.5-mini**, **Qwen2.5-3B** cần `adamw_bnb_8bit` + gradient checkpointing ở Stage 2 (đã cấu hình sẵn trong model configs).

---

## Cài đặt

### Yêu cầu phần cứng

| Kịch bản | GPU | Ghi chú |
|----------|-----|---------|
| Paper (full scale) | 8× A6000 48 GB | DeepSpeed Zero2 |
| Vi-LEO-MINI / LEO-MINI 3B / Phi | 1× A6000 / A100 48 GB | 8-bit Adam + gradient checkpointing |
| LEO-MINI 1B | 1× RTX 3090 24 GB | AdamW chuẩn |
| Quick eval (Colab) | T4 16 GB | 4-bit quantisation |

### Cài đặt môi trường

```bash
conda create -n leomini python=3.10 -y
conda activate leomini

# PyTorch 2.5.1 + CUDA 12.4
# Kiểm tra CUDA version: nvidia-smi | grep "CUDA Version"
# Thay cu124 → cu120/cu121 nếu driver chỉ hỗ trợ CUDA 12.0/12.1
pip install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "torchaudio==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

> **Lưu ý NumPy**: `requirements.txt` pin `numpy<2.0`. Nếu gặp `_ARRAY_API not found`, chạy `pip install "numpy<2"`.

### Xác thực Hugging Face

```bash
# Đăng nhập một lần, token lưu vào ~/.cache/huggingface/token
hf auth login
```

Một số model là **gated repos** — cần accept license tại trang model:
- Meta Llama: được approve ngay lập tức
- Qwen2.5: không cần gate

### Tải pretrained weights

```bash
# LLM backbone
hf download meta-llama/Llama-3.2-1B-Instruct   # LEO-MINI gốc
hf download meta-llama/Llama-3.2-3B-Instruct   # LEO-MINI gốc
hf download microsoft/Phi-3.5-mini-instruct     # LEO-MINI gốc
hf download Qwen/Qwen2.5-3B-Instruct            # Vi-LEO-MINI

# Vision experts (dùng cho cả hai variant)
hf download openai/clip-vit-large-patch14-336
hf download google/pix2struct-large
# EVA-02 và ConvNeXt: tải tự động qua open_clip (chỉ LEO-MINI gốc)

# EAGLE Stage-2 checkpoint (shortcut Stage 1+2, chỉ Llama3-8B)
hf download NVEagle/Eagle-X4-8B-Plus
```

### Tải training data

#### LEO-MINI gốc (EAGLE + LLaVA)

```bash
mkdir -p data/eagle data/eagle_sft

# Stage 1 — LLaVA-Pretrain 558K (alignment, ~12 GB)
hf download liuhaotian/LLaVA-Pretrain \
    --repo-type dataset \
    --local-dir data/eagle
unzip data/eagle/images.zip -d data/eagle/images

# Stage 2 — LLaVA-1.5 SFT 665K
hf download liuhaotian/LLaVA-Instruct-150K \
    --repo-type dataset \
    --local-dir data/eagle_sft
```

> **Ảnh Stage 2** đến từ nhiều nguồn (COCO, GQA, OCR-VQA, TextVQA, VisualGenome).  
> Xem [LLaVA Data Preparation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md). Đặt vào `data/eagle_sft/images/`.
>
> Flag `--repo-type dataset` là bắt buộc.

#### Vi-LEO-MINI (ViTextVQA — tải tự động)

Cả 3 bộ dữ liệu cho Vi-LEO-MINI (`KTVIC`, `OpenViVQA`, `ViTextVQA`) đều được tải **tự động qua HuggingFace `datasets`** khi chạy training lần đầu. Không cần tải thủ công.

```python
# Tương đương với điều trainer thực hiện:
from datasets import load_dataset
ds = load_dataset("ai-enthusiasm-community/KTVIC", split="train")
# Cache tự động tại ~/.cache/huggingface/datasets/
```

Nếu muốn cache về thư mục cụ thể:
```yaml
# configs/models/qwen2_5_3b_vi.yaml
vi_cache_dir: "/data/hf_cache"
```

---

## Training — LEO-MINI (gốc)

Model-specific configs nằm ở `configs/models/`. Mỗi file chỉ chứa các key khác với stage config mặc định.

```bash
# Stage 1 — Warmup Projector
bash scripts/run_stage1.sh configs/models/llama3_2_1b.yaml
bash scripts/run_stage1.sh configs/models/llama3_2_3b.yaml
bash scripts/run_stage1.sh configs/models/phi3_5_mini.yaml

# Stage 2 — Full SFT
bash scripts/run_stage2.sh configs/models/llama3_2_1b.yaml
bash scripts/run_stage2.sh configs/models/llama3_2_3b.yaml
bash scripts/run_stage2.sh configs/models/phi3_5_mini.yaml

# Stage 3 — Token Reduction
bash scripts/run_stage3.sh configs/models/llama3_2_1b.yaml
bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml
bash scripts/run_stage3.sh configs/models/phi3_5_mini.yaml

# Multi-GPU
NUM_GPUS=4 bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml
```

### Cấu hình model configs

#### Llama-3.2-1B (`configs/models/llama3_2_1b.yaml`)
```yaml
base_model_path: "meta-llama/Llama-3.2-1B-Instruct"
output_dir: "checkpoints/llama3_2_1b"
per_device_batch_size: 8
gradient_accumulation: 8
optim: "adamw_torch"
gradient_checkpointing: false
```

#### Llama-3.2-3B (`configs/models/llama3_2_3b.yaml`)
```yaml
base_model_path: "meta-llama/Llama-3.2-3B-Instruct"
output_dir: "checkpoints/llama3_2_3b"
per_device_batch_size: 2
gradient_accumulation: 32
optim: "adamw_bnb_8bit"      # bắt buộc để tránh OOM Stage 2
gradient_checkpointing: true
```

#### Phi-3.5-mini (`configs/models/phi3_5_mini.yaml`)
```yaml
base_model_path: "microsoft/Phi-3.5-mini-instruct"
output_dir: "checkpoints/phi3_5_mini"
per_device_batch_size: 2
gradient_accumulation: 32
optim: "adamw_bnb_8bit"
gradient_checkpointing: true
```

> **Quan trọng**: Dùng `base_model_path` (không phải `llm_path`) trong model configs.  
> Trainer tự động derive `llm_path` cho Stage 2/3 từ `output_dir/stage{n-1}/llm_checkpoint/`.

---

## Training — Vi-LEO-MINI (Vietnamese TextVQA)

### Chạy 3 stages

```bash
# Stage 1 — Projector warmup trên ViTextVQA train
bash scripts/run_vi_stage1.sh configs/models/qwen2_5_3b_vi.yaml

# Stage 2 — Full SFT trên ViTextVQA train (yêu cầu Stage 1 xong)
bash scripts/run_vi_stage2.sh configs/models/qwen2_5_3b_vi.yaml

# Stage 3 — CoTR + MMoE-LLM (yêu cầu Stage 2 xong)
bash scripts/run_vi_stage3.sh configs/models/qwen2_5_3b_vi.yaml

# Multi-GPU
NUM_GPUS=2 bash scripts/run_vi_stage3.sh configs/models/qwen2_5_3b_vi.yaml
```

Nếu Stage trước chưa chạy, trainer sẽ **thoát ngay với lỗi rõ ràng** (`sys.exit(1)`) — không silent fallback.

### Model config (`configs/models/qwen2_5_3b_vi.yaml`)

```yaml
base_model_path: "Qwen/Qwen2.5-3B-Instruct"
output_dir:      "checkpoints/qwen2_5_3b_vi"

# 2 vision experts chuyên cho text-image
vision_experts:
  - clip
  - pix2struct

n_visual:    128     # 128 tokens giữ detail chữ tiếng Việt
d_proj_cotr: 256
lora_rank:   16
num_special: 3

vi_hf_dataset: "minhquan6203/ViTextVQA"
vi_train_path: "train"
vi_val_path:   "validation"

log_dir: "logs/qwen2_5_3b_vi"

per_device_batch_size: 4
gradient_accumulation: 16
optim: "adamw_bnb_8bit"
gradient_checkpointing: true
```

### Checkpoint structure sau khi train xong

```
checkpoints/qwen2_5_3b_vi/
├── stage1/
│   ├── llm_checkpoint/        ← Qwen2.5-3B đã fine-tune projector warmup
│   └── projector_weights.pt   ← dùng làm projector_path cho Stage 2
├── stage2/
│   ├── llm_checkpoint/        ← Qwen2.5-3B đã full SFT
│   └── projector_weights.pt   ← dùng làm projector_path cho Stage 3
└── stage3/
    └── stage3_adapter_weights.pt  ← projector + CoTR + MMoE-LLM LoRA weights
```

### Qwen2.5 conversation format

Vi-LEO-MINI dùng Qwen2.5 chat template thay cho USER:/ASSISTANT: của bản gốc:

```
<|im_start|>system
Bạn là trợ lý AI thông minh, hãy trả lời câu hỏi dựa trên nội dung hình ảnh...<|im_end|>
<|im_start|>user
<image>
{câu hỏi tiếng Việt}<|im_end|>
<|im_start|>assistant
{câu trả lời}<|im_end|>
```

Label masking: tất cả tokens trước `<|im_start|>assistant\n` được mask bằng `-100`.

---

## Evaluation — LEO-MINI (gốc, 12 benchmarks)

### Đánh giá một model

```bash
python -m src.eval.evaluator \
    --model_path checkpoints/stage3 \
    --model_name "Llama-3.2-3B" \
    --tasks pope,scienceqa,textvqa,mme,mmmu \
    --output_dir results/llama3b
```

### So sánh ba model

```bash
BASE_1B="meta-llama/Llama-3.2-1B-Instruct" \
CKPT_1B="checkpoints/llama3_2_1b/stage3/stage3_adapter_weights.pt" \
BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
BASE_PHI="microsoft/Phi-3.5-mini-instruct" \
CKPT_PHI="checkpoints/phi3_5_mini/stage3/stage3_adapter_weights.pt" \
    bash scripts/run_benchmark_compare.sh
```

### 12 benchmarks đầy đủ

| Benchmark | Metric | Target (paper, 8B) |
|-----------|--------|-------------------|
| MME | Perception score | 1583.0 |
| MMBench | Overall acc | 77.0 |
| SEED-Bench | Overall acc | 75.8 |
| GQA | Exact match | 64.5 |
| ScienceQA | Acc | 84.5 |
| MMMU | Acc | 38.8 |
| POPE | Acc | 90.3 |
| AI2D | Acc | 75.7 |
| TextVQA | Acc | 75.1 |
| ChartQA | Relaxed acc | 80.5 |
| OCRBench | Acc | 62.4 |
| VizWiz | VQA acc | 69.3 |

---

## Evaluation — Vi-LEO-MINI (ViTextVQA, 3 metrics)

### Metrics

| Metric | Mô tả | Công thức |
|--------|-------|-----------|
| **ANLS** (main) | Average Normalized Levenshtein Similarity | `1 - NL` nếu `NL < 0.5`, else `0`; trung bình trên tất cả câu hỏi |
| **EM** | Exact Match (%) | `1` nếu dự đoán khớp chính xác với bất kỳ GT nào sau normalize |
| **F1** | Token-level F1 (%) | Bag-of-words overlap, max over GT answers |

> **Lưu ý**: Normalize tiếng Việt **giữ nguyên dấu thanh** (tone marks) vì chúng thay đổi nghĩa hoàn toàn (ma / má / mà / mả / mã / mạ). Chỉ lowercase + bỏ punctuation + collapse spaces.

### Chạy evaluation

```bash
# Sau khi hoàn thành Stage 3:
python -m src.eval.vi_evaluator \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --stage3_weights checkpoints/qwen2_5_3b_vi/stage3/stage3_adapter_weights.pt \
    --split test \
    --output_dir results/vi_leomini/ \
    --log_dir logs/qwen2_5_3b_vi/ \
    --vision_experts clip pix2struct \
    --n_visual 128

# Nhanh — chỉ 200 mẫu:
python -m src.eval.vi_evaluator \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --stage3_weights checkpoints/qwen2_5_3b_vi/stage3/stage3_adapter_weights.pt \
    --split validation \
    --limit 200 \
    --load_in_4bit \
    --output_dir results/vi_leomini/
```

Output file JSON được lưu tại `results/vi_leomini/eval_stage3_test_<timestamp>.json`:

```json
{
  "stage": 3,
  "split": "test",
  "anls": 0.6123,
  "em": 0.4891,
  "f1": 0.5934,
  "anls_pct": 61.23,
  "em_pct": 48.91,
  "f1_pct": 59.34,
  "n_samples": 2000,
  "per_sample": [
    {
      "idx": 0,
      "pred": "Phở Bắc",
      "gts": ["Phở Bắc", "PHỞ BẮC"],
      "anls": 1.0,
      "em": 1.0,
      "f1": 1.0
    },
    ...
  ]
}
```

---

## Logging & Visualization

### Structured JSON logs

Mỗi training run tự động ghi log vào `logs/<model>/<run_id>.json`:

```json
{
  "run_id": "Qwen2.5_3B_Instruct_stage1_20260623_143000",
  "config": { "stage": 1, "n_visual": 128, "vision_experts": ["clip", "pix2struct"], ... },
  "events": [
    { "type": "train_step", "stage": 1, "global_step": 10,  "loss": 2.314, "learning_rate": 9.8e-4 },
    { "type": "train_step", "stage": 1, "global_step": 20,  "loss": 1.982, "balance_loss": 0.021 },
    { "type": "eval",       "stage": 2, "global_step": 500, "split": "validation",
      "anls": 0.412, "em": 0.283, "f1": 0.471, "n_samples": 1000 },
    { "type": "stage_end",  "stage": 1, "duration_sec": 3600, "checkpoint_path": "..." }
  ]
}
```

### Export charts

```bash
# Tự động sau mỗi eval run
python -m src.utils.visualizer \
    --log logs/qwen2_5_3b_vi/run_id.json \
    --output results/vi_leomini/

# Charts được tạo ra trong results/vi_leomini/plots/:
#   training_loss_stage1.png   — loss curve Stage 1
#   training_loss_stage2.png   — loss curve Stage 2
#   training_loss_stage3.png   — loss curve Stage 3
#   balance_loss_stage3.png    — balance loss vs total loss (Stage 3)
#   eval_metrics_history.png   — ANLS / EM / F1 qua các checkpoints
#   learning_rate_schedule.png — LR schedule
#   anls_distribution_stage3_test.png  — histogram phân phối ANLS per-sample
```

---

## Demo và Quick Eval (Google Colab)

### Cài đặt trên Colab

```python
!pip install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "torchaudio==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124
!pip install "numpy<2" transformers peft bitsandbytes accelerate timm \
    open-clip-torch lmms-eval huggingface_hub[hf_xet] datasets matplotlib

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/leo_mini')
```

### Load Vi-LEO-MINI (4-bit, Colab T4)

```python
import torch
from src.models.leo_mini import LeoMini

model = LeoMini.from_pretrained(
    llm_path="Qwen/Qwen2.5-3B-Instruct",
    stage3_weights="/content/drive/MyDrive/vi_leomini/stage3_adapter_weights.pt",
    load_in_4bit=True,
    vision_experts=["clip", "pix2struct"],
    n_visual=128,
)
model.eval()
```

### Demo inference tiếng Việt

```python
from PIL import Image
from transformers import CLIPImageProcessor, AutoProcessor

image = Image.open("bien_hieu.jpg")   # ảnh biển hiệu tiếng Việt
question = "Tên cửa hàng trên biển hiệu là gì?"

# Visual preprocessing
clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
p2s_proc  = AutoProcessor.from_pretrained("google/pix2struct-large")

pixel_values = clip_proc(images=image, return_tensors="pt").pixel_values
p2s_inputs   = p2s_proc(images=image, text="", return_tensors="pt", max_patches=576)
pix2struct_inputs = {
    "flattened_patches": p2s_inputs.flattened_patches,
    "attention_mask":    p2s_inputs.attention_mask,
}

# Text preprocessing (Qwen2.5 chat format)
from src.data.vitextvqa_dataset import _qwen2_prompt_only
from src.models.leo_mini import IMAGE_TOKEN_INDEX

prompt = _qwen2_prompt_only(question)
enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = enc.input_ids
input_ids[input_ids == model.tokenizer.convert_tokens_to_ids("<image>")] = IMAGE_TOKEN_INDEX

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids.cuda(),
        pixel_values=pixel_values.cuda(),
        pix2struct_inputs={k: v.cuda() for k, v in pix2struct_inputs.items()},
        max_new_tokens=64,
        do_sample=False,
    )

answer = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Câu trả lời:", answer)
```

### Quick eval trên ViTextVQA validation (Colab)

```python
!python -m src.eval.vi_evaluator \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --stage3_weights /content/drive/MyDrive/vi_leomini/stage3_adapter_weights.pt \
    --split validation \
    --limit 500 \
    --load_in_4bit \
    --vision_experts clip pix2struct \
    --n_visual 128 \
    --output_dir /content/results
```

---

## Cấu trúc thư mục

```
leo_mini/
├── docs/
│   └── leomini.pdf                           # Bài báo gốc (EMNLP 2025)
├── src/
│   ├── models/
│   │   ├── cotr.py                           # CoTR: Eq.2–7 (token reduction)
│   │   ├── mmoe_llm.py                       # MMoE-LLM: LoRA experts + router + balance loss
│   │   ├── vision_experts.py                 # CLIP / EVA-02 / ConvNeXt / Pix2Struct
│   │   │                                     #   build_mmoe_vision(experts=["clip","pix2struct"])
│   │   ├── projector.py                      # Visual Projector (2-layer MLP)
│   │   └── leo_mini.py                       # Full pipeline + from_pretrained + set_stage()
│   ├── data/
│   │   ├── dataset.py                        # EAGLEDataset + LLaVADataset (gốc)
│   │   ├── collator.py                       # Batch collation
│   │   └── vitextvqa_dataset.py              # ViTextVQADataset (HF auto-download, Qwen2.5 format)
│   ├── train/
│   │   └── trainer.py                        # 3-stage trainer + Vi fields + logging callback
│   ├── eval/
│   │   ├── evaluator.py                      # lmms-eval wrapper + compare_models()
│   │   ├── lmms_adapter.py                   # lmms-eval @register_model adapter
│   │   ├── metrics.py                        # ANLS / EM / F1 (Vietnamese-aware)
│   │   └── vi_evaluator.py                   # ViTextVQAEvaluator + CLI
│   └── utils/
│       ├── logger.py                         # ViLeoMiniLogger (JSON) + LeoMiniLoggingCallback
│       └── visualizer.py                     # 5 chart types: loss, metrics, LR, ANLS dist
├── configs/
│   ├── config_stage1.yaml                    # LEO-MINI Stage 1 (gốc)
│   ├── config_stage2.yaml                    # LEO-MINI Stage 2 (gốc)
│   ├── config_stage3.yaml                    # LEO-MINI Stage 3 (gốc)
│   ├── config_vi_stage1.yaml                 # Vi-LEO-MINI Stage 1
│   ├── config_vi_stage2.yaml                 # Vi-LEO-MINI Stage 2
│   ├── config_vi_stage3.yaml                 # Vi-LEO-MINI Stage 3
│   ├── models/
│   │   ├── llama3_2_1b.yaml                  # Override: Llama-3.2-1B
│   │   ├── llama3_2_3b.yaml                  # Override: Llama-3.2-3B
│   │   ├── phi3_5_mini.yaml                  # Override: Phi-3.5-mini
│   │   └── qwen2_5_3b_vi.yaml                # Override: Qwen2.5-3B (Vi-LEO-MINI)
│   └── deepspeed_zero2.json                  # DeepSpeed config (multi-GPU)
├── scripts/
│   ├── run_stage1.sh                         # LEO-MINI gốc
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   ├── run_vi_stage1.sh                      # Vi-LEO-MINI
│   ├── run_vi_stage2.sh
│   ├── run_vi_stage3.sh
│   ├── run_benchmark_compare.sh              # so sánh LEO-MINI models
│   └── run_eval_leomini.py                   # wrapper cho lmms-eval
├── logs/                                     # JSON training logs (auto-created)
├── results/                                  # eval JSON + plots (auto-created)
├── requirements.txt
└── README.md
```

---

## Chi tiết kỹ thuật

### CoTR — Conditional Token Reduction (Section 3.2)

Với mỗi expert $i$ có $N_i$ visual tokens, feature dim $d_i^V$:

$$s_i^{\text{QUERY}} = \bar{Q}_i \bar{I}_i^\top \in \mathbb{R}^{N^V \times N_i} \tag{2}$$

$$s_i^{\text{SELF}} = \mathbf{1} \cdot \bar{I}_i \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{3}$$

$$s_i^{\text{CROSS}} = \sum_{j \in [m] \setminus \{i\}} \mathbf{1} \cdot \bar{I}_j \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{4}$$

$$s_i^{\text{TEXT}} = \mathbf{1} \cdot \hat{T} \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{5}$$

$$\alpha_i = \text{softmax}\!\left(\frac{s_i^{\text{QUERY}} + s_i^{\text{SELF}} + s_i^{\text{CROSS}} + s_i^{\text{TEXT}}}{\sqrt{d_i^V}}\right) \tag{6}$$

$$\bar{I}_i = \alpha_i \cdot I_i \in \mathbb{R}^{N^V \times d_i^V}, \qquad \bar{I} = \text{concat}([\bar{I}_1, \ldots, \bar{I}_m]) \tag{7}$$

Scale factor dùng $\sqrt{d_i^V}$ (dim gốc của expert, **không phải** d_proj).  
Vi-LEO-MINI: $m=2$, $N^V=128$, $d^V = 1024 + 2048 = 3072$.

### MMoE-LLM (Section 3.3)

Thay thế `down_proj` trong **mọi** MLP block của LLM:

$$y = f_{\text{ORI}}(x) + f_{\text{GEN}}(x) + \sum_{i \in E'} f_i^E(x) / k \tag{8}$$

- $f_{\text{ORI}}$: original frozen `down_proj`
- $f_{\text{GEN}}$: general LoRA adapter (rank=16, luôn active)
- $f_i^E$: $E=3$ special LoRA experts, top-$k=1$ được router chọn

**Balance loss** (tránh routing collapse):

$$\mathcal{L}_{\text{balance}} = \lambda \sum_{i=1}^{E} \left( f_i - \frac{1}{E} \right)^2, \quad \lambda = 0.05$$

Tổng loss = cross-entropy + $\mathcal{L}_{\text{balance}}$ (chỉ Stage 3).

---

## Kết quả kỳ vọng

### LEO-MINI gốc

| Metric | Paper (8B) | Llama-1B | Llama-3B | Phi-3.5 |
|--------|-----------|----------|----------|---------|
| MME | 1583.0 | 1380–1450 | 1480–1530 | 1510–1560 |
| POPE | 90.3 | 83–85 | 86–88 | 87–89 |
| ScienceQA | 84.5 | 77–79 | 80–82 | 81–83 |
| TextVQA | 75.1 | 66–69 | 70–72 | 71–73 |
| Visual Tokens | **64** | **64** | **64** | **64** |

### Vi-LEO-MINI (ViTextVQA)

| Metric | Qwen2.5-3B (Stage 2) | Qwen2.5-3B (Stage 3 + CoTR) |
|--------|---------------------|------------------------------|
| ANLS | (baseline) | kỳ vọng tăng với token reduction |
| EM (%) | — | — |
| F1 (%) | — | — |
| Visual Tokens | 576×2 = 1152 | **128** |

> Kết quả thực tế phụ thuộc vào dữ liệu training và số epoch. Chạy eval sau từng Stage để theo dõi tiến trình.

---

## Tham khảo

```bibtex
@inproceedings{wang2025leomini,
  title     = {LEO-MINI: An Efficient Multimodal Large Language Model
               using Conditional Token Reduction and Mixture of
               Multi-Modal Experts},
  author    = {Wang, Yimu and Nasr Azadani, Mozhgan and
               Sedwards, Sean and Czarnecki, Krzysztof},
  booktitle = {Proceedings of EMNLP 2025},
  pages     = {7246--7261},
  year      = {2025}
}
```

**Các công trình liên quan:**
- [EAGLE](https://github.com/shi-labs/eagle) — Multi-encoder baseline (Shi et al., 2024)
- [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) — Visual instruction tuning
- [ViTextVQA](https://huggingface.co/datasets/minhquan6203/ViTextVQA) — Vietnamese scene-text VQA dataset
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — Multilingual LLM backbone
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — Benchmark evaluation framework
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 8-bit Adam optimizer
