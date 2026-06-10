# LEO-MINI

Tái hiện thực nghiệm bài báo **"LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts"** (EMNLP 2025).

> Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki — University of Waterloo

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

```
Image(s) + Text Instruction
       │
       ├─ MMoE-Vision (4 experts — tất cả frozen trong Stage 3)
       │     ├─ CLIP ViT-L/14-336       → (B, 576, 1024)
       │     ├─ EVA-02 CLIP-L-14-336    → (B, 576, 1024)
       │     ├─ ConvNeXt-Large-D        → (B, 576,  768)
       │     └─ Pix2Struct-Large        → (B, 576, 2048)
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

### 3-Stage Training (Table 6, paper)

| Stage | Trainable | Frozen | Data |
|-------|-----------|--------|------|
| 1 — Warmup Projector | Visual Projector | LLM, Vision Experts | EAGLE alignment |
| 2 — Full SFT | Tất cả | — | EAGLE SFT |
| 3 — Token Reduction | **CoTR + MMoE-LLM + Projector** | LLM backbone, Vision Experts | LLaVA-v1.5 665K |

---

## Base Models được hỗ trợ

Repo này hỗ trợ ba LLM backbone thay thế cho cấu hình 1 GPU 48 GB:

| Model | VRAM Stage 2 | VRAM Stage 3 | Optimizer Stage 2 | Chất lượng kỳ vọng |
|-------|-------------|-------------|-------------------|--------------------|
| `meta-llama/Llama-3.2-1B-Instruct` | ~14 GB | ~8 GB | AdamW chuẩn | Thấp nhất, nhanh nhất |
| `meta-llama/Llama-3.2-3B-Instruct` | ~27 GB | ~14 GB | **8-bit Adam** | Trung bình |
| `microsoft/Phi-3.5-mini-instruct` | ~30 GB | ~16 GB | **8-bit Adam** | Tốt nhất trong nhóm nhỏ |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ~65 GB | ~32 GB | DeepSpeed Zero2 | **Paper** (8× GPU) |

> **Llama-3.2-3B** và **Phi-3.5-mini** cần `adamw_bnb_8bit` + gradient checkpointing ở Stage 2 (đã cấu hình sẵn trong model configs).

---

## Cài đặt

### Yêu cầu phần cứng

| Kịch bản | GPU | Ghi chú |
|----------|-----|---------|
| Paper (full scale) | 8× A6000 48 GB | DeepSpeed Zero2 |
| Demo / đồ án (3B, Phi) | 1× A6000 / A100 48 GB | 8-bit Adam + gradient checkpointing |
| Demo / đồ án (1B) | 1× RTX 3090 24 GB | AdamW chuẩn |
| Quick eval (Colab) | T4 16 GB | 4-bit quantisation |

### Cài đặt môi trường

```bash
conda create -n leomini python=3.10 -y
conda activate leomini

# PyTorch 2.4+ + CUDA 12.1
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Tải pretrained weights

```bash
# LLM backbone — chọn 1 trong các backbone sau:
hf download meta-llama/Llama-3.2-1B-Instruct   # 1B (nhẹ nhất)
hf download meta-llama/Llama-3.2-3B-Instruct   # 3B (cân bằng)
hf download microsoft/Phi-3.5-mini-instruct     # Phi (hiệu năng tốt)
hf download meta-llama/Meta-Llama-3-8B-Instruct # 8B (paper gốc)

# Vision experts (tất cả các variant đều dùng chung)
hf download openai/clip-vit-large-patch14-336
hf download google/pix2struct-large
# EVA-02 CLIP-L-14-336: tải tự động qua open_clip khi khởi tạo (QuanSun/EVA-CLIP)
# ConvNeXt-Large-D: tải tự động qua open_clip khi khởi tạo

# EAGLE Stage-2 checkpoint (dùng làm điểm khởi đầu Stage 3, thay thế train Stage 1+2)
hf download NVEagle/Eagle-X4-8B-Plus
```

---

## Training (GPU Server)

### Chạy nhanh với model configs

Model-specific configs nằm ở `configs/models/`. Mỗi file chỉ chứa các key khác với stage config mặc định (llm_path, batch size, optimizer, v.v.).

```bash
# Stage 1 — Warmup Projector
bash scripts/run_stage1.sh configs/models/llama3_2_1b.yaml   # Llama-3.2-1B
bash scripts/run_stage1.sh configs/models/llama3_2_3b.yaml   # Llama-3.2-3B
bash scripts/run_stage1.sh configs/models/phi3_5_mini.yaml   # Phi-3.5-mini

# Stage 2 — Full SFT (3B/Phi dùng 8-bit Adam + gradient checkpointing, cấu hình sẵn)
bash scripts/run_stage2.sh configs/models/llama3_2_1b.yaml   # Llama-3.2-1B
bash scripts/run_stage2.sh configs/models/llama3_2_3b.yaml   # Llama-3.2-3B
bash scripts/run_stage2.sh configs/models/phi3_5_mini.yaml   # Phi-3.5-mini

# Stage 3 — Token Reduction (CoTR + MMoE-LLM + Projector)
bash scripts/run_stage3.sh configs/models/llama3_2_1b.yaml   # Llama-3.2-1B
bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml   # Llama-3.2-3B
bash scripts/run_stage3.sh configs/models/phi3_5_mini.yaml   # Phi-3.5-mini
```

Để train song song nhiều GPU:
```bash
NUM_GPUS=4 bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml
```

### Cấu hình chi tiết từng model

#### Llama-3.2-1B (`configs/models/llama3_2_1b.yaml`)
```yaml
base_model_path: "meta-llama/Llama-3.2-1B-Instruct"   # HuggingFace ID cho Stage 1
output_dir: "checkpoints/llama3_2_1b"                  # cách ly checkpoint giữa các models
per_device_batch_size: 8
gradient_accumulation: 8
optim: "adamw_torch"
gradient_checkpointing: false
deepspeed: null
```

#### Llama-3.2-3B (`configs/models/llama3_2_3b.yaml`)
```yaml
base_model_path: "meta-llama/Llama-3.2-3B-Instruct"
output_dir: "checkpoints/llama3_2_3b"
per_device_batch_size: 2
gradient_accumulation: 32          # effective batch = 64
optim: "adamw_bnb_8bit"            # BẮTBUỘC để tránh OOM
gradient_checkpointing: true
deepspeed: null
```

#### Phi-3.5-mini (`configs/models/phi3_5_mini.yaml`)
```yaml
base_model_path: "microsoft/Phi-3.5-mini-instruct"
output_dir: "checkpoints/phi3_5_mini"
per_device_batch_size: 2
gradient_accumulation: 32
optim: "adamw_bnb_8bit"
gradient_checkpointing: true
deepspeed: null
```

> **Lưu ý quan trọng**: Dùng `base_model_path` (không phải `llm_path`) trong model configs.
> Trainer tự động derive `llm_path` cho Stage 2/3 từ `output_dir/stage{n-1}/llm_checkpoint/`.
> Nếu dùng `llm_path` trong model config, Stage 2/3 sẽ load lại base model thay vì checkpoint Stage trước.

### Cấu hình Stage 3 mặc định (`configs/config_stage3.yaml`)
```yaml
n_visual:            64        # N^V: 576 → 64 tokens qua CoTR
d_proj_cotr:         256       # chiều projection chung trong CoTR
lora_rank:           16        # LoRA rank cho tất cả experts
num_special:         3         # E=3 special LoRA experts
balance_loss_lambda: 0.05      # λ cho L_balance
learning_rate:       2.0e-5
per_device_batch_size: 4
gradient_accumulation: 16      # effective batch = 64
```

---

## Evaluation

### Đánh giá một model

```bash
python -m src.eval.evaluator \
    --model_path checkpoints/stage3 \
    --model_name "Llama-3.2-3B" \
    --tasks pope,scienceqa,textvqa,mme,mmmu \
    --output_dir results/llama3b
```

### So sánh ba model (Llama-1B vs Llama-3B vs Phi)

```bash
# Nhanh — 200 mẫu/task, 4-bit quant (chỉ 3B)
BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
TASKS="pope,scienceqa,textvqa" LIMIT=200 LOAD_4BIT=1 \
    bash scripts/run_benchmark_compare.sh

# Full benchmark — cả 3 models
BASE_1B="meta-llama/Llama-3.2-1B-Instruct" \
CKPT_1B="checkpoints/llama3_2_1b/stage3/stage3_adapter_weights.pt" \
BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
BASE_PHI="microsoft/Phi-3.5-mini-instruct" \
CKPT_PHI="checkpoints/phi3_5_mini/stage3/stage3_adapter_weights.pt" \
    bash scripts/run_benchmark_compare.sh
```

Output mẫu:
```
============================================================
  BENCHMARK COMPARISON
============================================================
  Task          Paper(8B)    Llama-1B    Llama-3B    Phi-3.5
------------------------------------------------------------
  pope              90.3        85.x        87.x        88.x
  scienceqa         84.5        79.x        81.x        82.x
  textvqa           75.1        68.x        71.x        72.x
  mme             1583.0      1420.x      1510.x      1530.x
  mmmu              38.8        32.x        35.x        36.x
------------------------------------------------------------
```

### Ablation — số lượng visual tokens (Table 2 trong paper)

```bash
python -m src.eval.evaluator \
    --model_path checkpoints/stage3 \
    --tasks mme,pope,textvqa,scienceqa \
    --ablation
# So sánh N^V ∈ {1, 16, 64, 256}
```

### 12 benchmarks đầy đủ theo paper

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

## Demo và Quick Eval (Google Colab)

Sau khi training, upload `checkpoints/stage3/stage3_adapter_weights.pt` lên Google Drive, rồi dùng Colab để load và demo.

### Cài đặt trên Colab

```python
!pip install transformers peft bitsandbytes accelerate timm open-clip-torch lmms-eval huggingface_hub[hf_xet]

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/leo_mini')  # clone repo vào /content/leo_mini
```

### Load model (Colab Free T4 — 4-bit)

```python
import torch
from src.models.leo_mini import LeoMini

model = LeoMini.from_pretrained(
    llm_path="meta-llama/Llama-3.2-3B-Instruct",    # hoặc Phi-3.5-mini
    stage3_weights="/content/drive/MyDrive/leomini/stage3_adapter_weights.pt",
    load_in_4bit=True,
)
model.eval()
```

### Demo inference

```python
from PIL import Image
import torch

image = Image.open("test.jpg")
question = "What is shown in the image?"

# Chuẩn bị visual input
processor = model.vision.experts[0].processor  # CLIPImageProcessor
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Chuẩn bị text input
input_text = f"<image>\nUSER: {question} ASSISTANT:"
enc = model.tokenizer(input_text, return_tensors="pt")
input_ids = enc.input_ids
input_ids[input_ids == model.tokenizer.convert_tokens_to_ids("<image>")] = -200

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids.cuda(),
        pixel_values=pixel_values.cuda(),
        max_new_tokens=256,
        do_sample=False,
    )

answer = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(answer)
```

### Quick eval (~20 phút trên T4, 500 mẫu/task)

```python
!python -m src.eval.evaluator \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --model_name "LEO-MINI-3B" \
    --tasks pope,textvqa,scienceqa \
    --limit 500 \
    --load_in_4bit \
    --output_dir /content/results
```

---

## Cấu trúc thư mục

```
leo_mini/
├── docs/
│   └── leomini.pdf                          # Bài báo gốc (EMNLP 2025)
├── src/
│   ├── models/
│   │   ├── cotr.py                          # CoTR: Eq.2–7 (token reduction)
│   │   ├── mmoe_llm.py                      # MMoE-LLM: LoRA experts + router + balance loss
│   │   ├── vision_experts.py                # CLIP / EVA-02 / ConvNeXt / Pix2Struct
│   │   ├── projector.py                     # Visual Projector (2-layer MLP)
│   │   └── leo_mini.py                      # Full pipeline + from_pretrained + set_stage()
│   ├── data/
│   │   ├── dataset.py                       # EAGLEDataset + LLaVADataset
│   │   └── collator.py                      # Batch collation
│   ├── train/
│   │   └── trainer.py                       # 3-stage trainer (HuggingFace Trainer)
│   └── eval/
│       ├── evaluator.py                     # lmms-eval wrapper + compare_models()
│       └── lmms_adapter.py                  # lmms-eval @register_model adapter
├── configs/
│   ├── config_stage1.yaml                   # Stage 1: warmup projector
│   ├── config_stage2.yaml                   # Stage 2: full SFT
│   ├── config_stage3.yaml                   # Stage 3: CoTR + MMoE-LLM
│   ├── models/
│   │   ├── llama3_2_1b.yaml                 # Override: Llama-3.2-1B
│   │   ├── llama3_2_3b.yaml                 # Override: Llama-3.2-3B (8-bit Adam)
│   │   └── phi3_5_mini.yaml                 # Override: Phi-3.5-mini (8-bit Adam)
│   └── deepspeed_zero2.json                 # DeepSpeed config (multi-GPU)
├── scripts/
│   ├── run_stage1.sh                        # usage: bash run_stage1.sh [model_config]
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   ├── run_benchmark_compare.sh             # so sánh 3 model cùng lúc
│   └── run_eval_leomini.py                  # wrapper để lmms-eval nhận diện model
├── requirements.txt
└── README.md
```

---

## Chi tiết kỹ thuật

### CoTR — Conditional Token Reduction (Section 3.2)

Với mỗi expert $i$ có $N_i$ visual tokens, feature dim $d_i^V$, và $N^V = 64$ output tokens:

$$s_i^{\text{QUERY}} = \bar{Q}_i \bar{I}_i^\top \in \mathbb{R}^{N^V \times N_i} \tag{2}$$

$$s_i^{\text{SELF}} = \mathbf{1} \cdot \bar{I}_i \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{3}$$

$$s_i^{\text{CROSS}} = \sum_{j \in [m] \setminus \{i\}} \mathbf{1} \cdot \bar{I}_j \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{4}$$

$$s_i^{\text{TEXT}} = \mathbf{1} \cdot \hat{T} \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i} \tag{5}$$

$$\alpha_i = \text{softmax}\!\left(\frac{s_i^{\text{QUERY}} + s_i^{\text{SELF}} + s_i^{\text{CROSS}} + s_i^{\text{TEXT}}}{\sqrt{d_i^V}}\right) \tag{6}$$

$$\bar{I}_i = \alpha_i \cdot I_i \in \mathbb{R}^{N^V \times d_i^V}, \qquad \bar{I} = \text{concat}([\bar{I}_1, \ldots, \bar{I}_m]) \tag{7}$$

Trong đó $\mathbf{1}$ là vector ones → tổng tất cả token positions (sum-pooling). Scale factor dùng $\sqrt{d_i^V}$ (feature dim gốc của expert, **không phải** d_proj).

### MMoE-LLM (Section 3.3)

Thay thế `down_proj` trong **mọi** MLP block của LLM:

$$y = f_{\text{ORI}}(x) + f_{\text{GEN}}(x) + \sum_{i \in E'} f_i^E(x) / k \tag{8}$$

- $f_{\text{ORI}}$: original frozen `down_proj` weight
- $f_{\text{GEN}}$: general LoRA adapter (rank=16, **luôn active**)
- $f_i^E$: $E=3$ special LoRA experts, top-$k=1$ được chọn bởi router
- Router: $R = \text{softmax}(f^{\text{ROUTING}}(\bar{I}, T, x)) \in \mathbb{R}^E$
  - 2-layer MLP + GELU
  - Input: hidden state $x$ ⊕ global visual context ⊕ global text context

**Balanced loss** (tránh routing collapse):

$$\mathcal{L}_{\text{balance}} = \lambda \sum_{i=1}^{E} \left( f_i - \frac{1}{E} \right)^2, \quad \lambda = 0.05$$

Tổng loss = cross-entropy + $\mathcal{L}_{\text{balance}}$ (chỉ ở Stage 3).

---

## Kết quả kỳ vọng

| Metric | Paper (8B) | Llama-1B (ước tính) | Llama-3B (ước tính) | Phi-3.5 (ước tính) |
|--------|-----------|---------------------|---------------------|---------------------|
| MME | 1583.0 | 1380–1450 | 1480–1530 | 1510–1560 |
| POPE | 90.3 | 83–85 | 86–88 | 87–89 |
| ScienceQA | 84.5 | 77–79 | 80–82 | 81–83 |
| TextVQA | 75.1 | 66–69 | 70–72 | 71–73 |
| Visual Tokens | **64** | **64** | **64** | **64** |

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
- [EAGLE](https://github.com/shi-labs/eagle) — Multi-encoder baseline (Shi et al., 2024); dùng `NVEagle/Eagle-X4-8B-Plus` làm Stage-2 checkpoint thay thế
- [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) — Visual instruction tuning
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — Benchmark evaluation framework
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 8-bit Adam optimizer cho Stage 2 trên single GPU
