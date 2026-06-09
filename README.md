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
       ├─ MMoE-Vision (4 experts)
       │     ├─ CLIP ViT-L/14-336       → (B, 576, 1024)
       │     ├─ EVA-02 CLIP-L-14-336    → (B, 576, 1024)
       │     ├─ ConvNeXt-Large-D        → (B, 576,  768)
       │     └─ Pix2Struct-Large        → (B, 576, 2048)
       │
       ├─ CoTR  [Stage 3]  576×4 → 64 tokens per expert
       │     Eq.2  s_QUERY  = Q̄_i · Ī_i^T
       │     Eq.3  s_SELF   = mean(Ī_i) · Ī_i^T
       │     Eq.4  s_CROSS  = Σ_{j≠i} mean(Ī_j) · Ī_i^T
       │     Eq.5  s_TEXT   = mean(T̂) · Ī_i^T
       │     Eq.6  α_i = softmax( (Σ scores) / √d_proj )
       │     Eq.7  Ī_i = α_i · I_i   →   concat → (B, 64, 4864)
       │
       ├─ Visual Projector  (B, 64, 4864) → (B, 64, 4096)
       │     Linear → GELU → Linear
       │
       └─ LLM + MMoE-LLM [Stage 3]
             concat([Ī_proj, T_embed]) → LLM
             down_proj replaced:
               y = f_ORI(x) + f_GEN(x) + f_top1(x)   (Eq.9)
             Router: 2-layer MLP + GELU ← (Ī, T, x)
```

### 3-Stage Training

| Stage | Trainable | Frozen | Data |
|-------|-----------|--------|------|
| 1 — Warmup Projector | Visual Projector | LLM, Vision Experts | EAGLE alignment |
| 2 — Full SFT | Tất cả | — | EAGLE SFT |
| 3 — Token Reduction | **CoTR + MMoE-LLM** | LLM, Vision Experts | LLaVA-v1.5 665K |

> **Trên GPU server:** Chạy cả 3 stages.  
> **Trên Colab:** Load checkpoint Stage 3, chạy demo + quick eval.

---

## Cài đặt

### Yêu cầu phần cứng

| Cấu hình (paper) | Cấu hình khuyến nghị |
|------------------|----------------------|
| 8× A6000 (48 GB) | 4–8× A100/A6000/H100 (≥ 40 GB) |
| DeepSpeed Zero2 | DeepSpeed Zero2 hoặc Zero3 |

### Cài đặt môi trường

```bash
conda create -n leomini python=3.10 -y
conda activate leomini

# PyTorch 2.1 + CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Tải pretrained weights

```bash
# LLM backbone
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
huggingface-cli download lmsys/vicuna-7b-v1.5  # nếu dùng variant Vicuna

# Vision experts
huggingface-cli download openai/clip-vit-large-patch14-336
huggingface-cli download Yuxin-CV/EVA-02-CLIP-L-14-336
huggingface-cli download google/pix2struct-large
# ConvNeXt tải tự động qua open_clip khi khởi tạo

# EAGLE Stage 1+2 checkpoint (dùng làm khởi điểm Stage 3)
huggingface-cli download shi-labs/eagle
```

---

## Training (GPU Server)

### Stage 1 — Warmup Visual Projector

```bash
# Cấu hình đường dẫn data trong configs/config_stage1.yaml
# eagle_data_path: "data/eagle/alignment.json"
# image_dir: "data/images"

NUM_GPUS=8 bash scripts/run_stage1.sh
# Thời gian ước tính: ~4–8 giờ trên 8×A100
```

### Stage 2 — Full Supervised Fine-tuning

```bash
# Cập nhật llm_path trong config_stage2.yaml → checkpoint Stage 1
NUM_GPUS=8 bash scripts/run_stage2.sh
# Thời gian ước tính: ~1–2 ngày trên 8×A100
```

### Stage 3 — Token Reduction Fine-tuning *(điểm cốt lõi)*

```bash
# Bắt đầu từ EAGLE checkpoint (bỏ qua Stage 1+2)
LLM_PATH="shi-labs/eagle" NUM_GPUS=8 bash scripts/run_stage3.sh

# Hoặc từ Stage 2 checkpoint
NUM_GPUS=8 bash scripts/run_stage3.sh
# Thời gian ước tính: ~8–16 giờ trên 8×A100
```

**Cấu hình Stage 3 mặc định** (`configs/config_stage3.yaml`):

```yaml
n_visual:             64       # N^V output tokens
lora_rank:            16       # LoRA rank cho tất cả experts
num_special:          3        # E=3 special experts
balance_loss_lambda:  0.05     # λ cho L_balance
learning_rate:        2.0e-5
per_device_batch_size: 8
gradient_accumulation: 8
```

---

## Evaluation (GPU Server)

### Full evaluation — 12 benchmarks theo paper

```bash
MODEL_PATH="checkpoints/stage3/stage3_adapter_weights.pt" \
bash scripts/run_eval.sh
```

| Benchmark | Loại | Target (paper) |
|-----------|------|----------------|
| MME | Perception | 1583.0 |
| MMBench | General | 77.0 |
| SEED-Bench | General | 75.8 |
| GQA | Visual Reasoning | 64.5 |
| ScienceQA | Knowledge | 84.5 |
| MMMU | Multi-discipline | 38.8 |
| POPE | Hallucination | 90.3 |
| AI2D | Diagram | 75.7 |
| TextVQA | OCR | 75.1 |
| ChartQA | Chart | 80.5 |
| OCRBench | OCR | 62.4 |
| VizWiz | VQA | 69.3 |

### Ablation — số lượng visual tokens (Table 2 trong paper)

```bash
MODEL_PATH="checkpoints/stage3/stage3_adapter_weights.pt" \
TASKS="mme,pope,textvqa,scienceqa" \
bash scripts/run_eval.sh --ablation
# So sánh N^V ∈ {1, 16, 64, 256}
```

---

## Demo và Quick Eval (Google Colab)

Sau khi training trên GPU server, upload `stage3_adapter_weights.pt` lên Google Drive, sau đó dùng Colab để load và demo:

### Colab Free (T4 16 GB) — 4-bit quantisation

```python
# Cài đặt
!pip install transformers peft bitsandbytes accelerate timm open-clip-torch

from google.colab import drive
drive.mount('/content/drive')

from src.models.leo_mini import LeoMini
model = LeoMini.from_pretrained(
    llm_path="meta-llama/Meta-Llama-3-8B-Instruct",
    stage3_weights="/content/drive/MyDrive/leomini/stage3_adapter_weights.pt",
    load_in_4bit=True,
)
model.eval()
```

### Demo inference

```python
from PIL import Image

image = Image.open("test.jpg")
question = "What number is shown on the bus?"

# Chuẩn bị input
processor = model.vision.experts[0].processor  # CLIPImageProcessor
pixel_values = processor(images=image, return_tensors="pt").pixel_values

input_text = f"<image>\nUSER: {question} ASSISTANT:"
input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids
input_ids[input_ids == model.tokenizer.convert_tokens_to_ids("<image>")] = -200

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids.cuda(),
        pixel_values=pixel_values.cuda(),
        max_new_tokens=256,
    )

print(model.tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Quick eval (500 mẫu, ~20 phút trên T4)

```python
!python -m src.eval.evaluator \
    --model_path /content/drive/MyDrive/leomini/stage3_adapter_weights.pt \
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
│   └── leomini.pdf                     # Bài báo gốc (EMNLP 2025)
├── src/
│   ├── models/
│   │   ├── cotr.py                     # CoTR: Eq.2–7
│   │   ├── mmoe_llm.py                 # MMoE-LLM: LoRA experts + router
│   │   ├── vision_experts.py           # CLIP / EVA-02 / ConvNeXt / Pix2Struct
│   │   ├── projector.py                # Visual Projector (2-layer MLP)
│   │   └── leo_mini.py                 # Full pipeline
│   ├── data/
│   │   ├── dataset.py                  # EAGLE + LLaVA-v1.5 loaders
│   │   └── collator.py                 # Batch collation
│   ├── train/
│   │   └── trainer.py                  # 3-stage trainer (DeepSpeed Zero2)
│   └── eval/
│       └── evaluator.py                # lmms-eval wrapper (12 benchmarks)
├── configs/
│   ├── config_stage1.yaml
│   ├── config_stage2.yaml
│   ├── config_stage3.yaml
│   └── deepspeed_zero2.json
├── scripts/
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   └── run_eval.sh
├── requirements.txt
└── PLAN.md                             # Kế hoạch thực nghiệm chi tiết
```

---

## Chi tiết kỹ thuật

### CoTR — Conditional Token Reduction

Với mỗi expert $i$ có $N_i$ visual tokens và feature dim $d_i$:

$$s_i^{\text{QUERY}} = \bar{Q}_i \bar{I}_i^\top \in \mathbb{R}^{N^V \times N_i}$$

$$s_i^{\text{SELF}} = \mathbf{1} \cdot \bar{I}_i \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i}$$

$$s_i^{\text{CROSS}} = \sum_{j \neq i} \mathbf{1} \cdot \bar{I}_j \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i}$$

$$s_i^{\text{TEXT}} = \mathbf{1} \cdot \hat{T} \bar{I}_i^\top \in \mathbb{R}^{1 \times N_i}$$

$$\alpha_i = \text{softmax}\!\left(\frac{s_i^{\text{QUERY}} + s_i^{\text{SELF}} + s_i^{\text{CROSS}} + s_i^{\text{TEXT}}}{\sqrt{d_{\text{proj}}}}\right)$$

$$\bar{I}_i = \alpha_i \cdot I_i \in \mathbb{R}^{N^V \times d_i}, \quad \bar{I} = \text{concat}([\bar{I}_1, \ldots, \bar{I}_m])$$

Học $\mathbf{1}$ theo nghĩa mean-pooling: $\mathbf{1} \cdot X \triangleq \text{mean}(X, \dim=0)$.

### MMoE-LLM

$$y = f_{\text{ORI}}(x) + f_{\text{GEN}}(x) + \sum_{i \in E'} f_i^E(x) / k$$

- $f_{\text{ORI}}$: frozen original `down_proj`
- $f_{\text{GEN}}$: general LoRA (rank=16, luôn active)
- $f_i^E$: 3 special LoRA experts, top-1 được chọn bởi router
- Router: 2-layer MLP + GELU, input = $({\bar{I}}_{\text{global}}, T_{\text{global}}, x)$

**Balanced loss** (tránh routing collapse):

$$\mathcal{L}_{\text{balance}} = \lambda \sum_i \left( f_i - \frac{1}{E} \right)^2, \quad \lambda = 0.05$$

---

## Kết quả kỳ vọng

| Metric | Paper (Llama3-8B) | Target thực nghiệm |
|--------|-------------------|--------------------|
| MME | 1583.0 | ≥ 1550 |
| MMBench | 77.0 | ≥ 76.0 |
| SEED | 75.8 | ≥ 75.0 |
| GQA | 64.5 | ≥ 63.0 |
| SQA | 84.5 | ≥ 83.0 |
| TextVQA | 75.1 | ≥ 74.0 |
| MMMU | 38.8 | ≥ 37.0 |
| POPE | 90.3 | ≥ 89.0 |
| AI2D | 75.7 | ≥ 74.0 |
| ChartQA | 80.5 | ≥ 79.0 |
| OCRBench | 62.4 | ≥ 61.0 |
| Visual Tokens | **64** | **64** |

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
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — Benchmark evaluation framework
