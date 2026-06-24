# CLAUDE.md — Vi-LEO-MINI Project Context

> Đọc file này trước mỗi session làm việc với repo này.
> Cập nhật khi có thay đổi kiến trúc, fix quan trọng, hoặc quyết định thiết kế mới.

---

## 1. Mục tiêu dự án

**Đề tài**: "Hệ thống hỏi đáp ảnh có chữ tiếng Việt áp dụng mô hình LEO-MINI"

Tái hiện và thích nghi bài báo **LEO-MINI** (EMNLP 2025):

> "LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token
> Reduction and Mixture of Multi-Modal Experts"
> — Wang et al., University of Waterloo

Mục tiêu cụ thể: huấn luyện và đánh giá hệ thống VQA tiếng Việt cho ảnh có chữ trên **1 GPU 48 GB**,
với các điều chỉnh phù hợp cho tiếng Việt:

- LLM backbone: **Qwen2.5-3B-Instruct** (thay vì Llama3-8B)
- Vision experts: **CLIP + Pix2Struct** (2 experts thay vì 4)
- N^V = **128** (thay vì 64 — cần chi tiết hơn cho ảnh chứa chữ)
- Dữ liệu huấn luyện thuần tiếng Việt (KTVIC → OpenViVQA → ViTextVQA)
- Metric chính: **ANLS**, metric phụ: **EM**, **F1**

Paper PDF: `docs/leomini.pdf`

---

## 2. Hai đóng góp cốt lõi của bài báo

### 2.1 CoTR — Conditional Token Reduction (Section 3.2, Eq. 2–7)

Giảm mỗi expert từ 576 tokens → 128 tokens bằng weighted aggregation với 4 loại attention:

```
Eq.2  s_QUERY  = Q̄_i · Ī_i^T          ∈ R^{N^V × N_i}   (query-visual)
Eq.3  s_SELF   = 1 · Ī_i · Ī_i^T       ∈ R^{1 × N_i}     (self-similarity)
Eq.4  s_CROSS  = Σ_{j≠i} 1·Ī_j·Ī_i^T  ∈ R^{1 × N_i}     (cross-expert)
Eq.5  s_TEXT   = 1 · T̂ · Ī_i^T         ∈ R^{1 × N_i}     (text-visual)
Eq.6  α_i = softmax((Σ scores) / √d_i^V)                  (scale bởi DIM GỐC)
Eq.7  Ī_i = α_i · I_i  →  cat([Ī_1,...,Ī_m])             (aggregation original space)
```

**Quan trọng**: "1" là ones-vector → sum (không phải mean).
Scale factor dùng `expert_dims[i]` (feature dim gốc), không phải `d_proj`.
**Vi-LEO-MINI dùng N^V = 128** (tăng từ 64 trong paper gốc để bắt được chữ nhỏ trong ảnh).

### 2.2 MMoE-LLM (Section 3.3, Eq. 8)

Thay thế `down_proj` trong TỪNG MLP block của LLM:

```
y = f_ORI(x) + f_GEN(x) + f_top1(x)            (Eq. 8, k=1)
```

- `f_ORI`: original frozen `down_proj`
- `f_GEN`: general LoRA expert (luôn active, rank=16)
- `f_top1`: special LoRA expert được Router chọn (top-k=1)
- Router: 2-layer MLP + GELU, input = `(Ī_global, T_global, x)`

**Balanced loss** (tránh routing collapse):
```
L_balance = λ · Σ_i (fraction_i − 1/E)²,  λ=0.05
```

Loss tổng = cross-entropy + L_balance (chỉ ở Stage 3).

---

## 3. Kiến trúc Vi-LEO-MINI

```
Image(s) + Text
    │
    ├─ MMoE-Vision (2 experts, frozen trong Stage 3)
    │   ├─ CLIP ViT-L/14-336     → (B, 576, 1024)   ← general visual
    │   └─ Pix2Struct-Large      → (B, 576, 2048)   ← OCR/text specialist
    │   Tổng d^V = 3072  (không dùng EVA-02 và ConvNeXt)
    │
    ├─ CoTR [Stage 3 only]  → (B, 128, 3072)
    │   Stage 1+2: simple cat → (B, 576, 3072)
    │
    ├─ VisualProjector  2-layer MLP (d^V → 4·d_LLM → d_LLM)
    │   → (B, N^V, d_LLM)  (N^V=128 Stage 3, 576 Stage 1+2, d_LLM=2048)
    │
    └─ Qwen2.5-3B-Instruct + MMoE-LLM [Stage 3 only]
        concat([visual_tokens, text_tokens]) → autoregressive generation
        IMAGE_TOKEN_INDEX = -200 (sentinel trong input_ids)
```

### Hyperparameters (Vi-LEO-MINI)

| Param | Value | Ghi chú |
|---|---|---|
| N^V (visual tokens sau CoTR) | **128** | Tăng từ 64 để bắt chữ nhỏ |
| d_proj_cotr | 256 | Giống paper gốc |
| lora_rank | 16 | Giống paper gốc |
| num_special (E) | 3 | Giống paper gốc |
| balance_loss_lambda | 0.05 | Giống paper gốc |
| d^V (visual feature dim) | 3072 | CLIP(1024) + Pix2Struct(2048) |
| d_LLM | 2048 | Qwen2.5-3B hidden size |

---

## 4. 3-Stage Training với dữ liệu tiếng Việt

| Stage | Trainable | Frozen | Dataset | HF ID |
|---|---|---|---|---|
| 1 Projector Warmup | VisualProjector | LLM, Vision Experts | KTVIC (captioning) | `ai-enthusiasm-community/KTVIC` |
| 2 Full SFT | Tất cả | — | OpenViVQA (general VQA) | `uitnlp/OpenViVQA-dataset` |
| 3 Token Reduction | CoTR + MMoE-LLM LoRAs + Routers + Projector | LLM backbone, Vision Experts | ViTextVQA (scene-text VQA) | `minhquan6203/ViTextVQA` |

**Dataset class mapping** (trong `src/data/vitextvqa_dataset.py`):
- Stage 1 → `KTVICDataset`  — parses `captions`/`caption_vi`/`segment_caption_vi`
- Stage 2 → `OpenViVQADataset` — parses `question`/`answer`, có custom JSON loader do cấu trúc đặc biệt
- Stage 3 → `ViTextVQADataset` — parses `question`/`answers` (multi-answer list)

**Kích hoạt Vi dataset path**: set `vi_train_path: "train"` trong model config.
Trainer tự chọn đúng `DatasetClass` và `hf_dataset_name` theo stage.

**Lưu ý Stage 3**: Projector cũng trainable trong Stage 3 (paper Table 6).

---

## 5. Evaluation Metrics

### 5.1 Metric chính: ANLS (Average Normalized Levenshtein Similarity)

```python
# src/eval/metrics.py — hàm anls_score()
NL = edit_distance(pred, gt) / max(len(pred), len(gt))
score = 1 - NL  if NL < 0.5  else 0.0
ANLS = mean(max_over_gt(score_i))   # trung bình qua toàn dataset
```

Threshold = 0.5 (chuẩn TextVQA).

### 5.2 Metric phụ: EM và F1

- **EM (Exact Match)**: 1.0 nếu normalized prediction khớp chính xác với bất kỳ GT nào
- **F1**: Token-level bag-of-words overlap (max over GT answers)

### 5.3 Vietnamese normalization

`normalize_vi()` trong `metrics.py`:
- Lowercase + bỏ dấu câu
- **GIỮ NGUYÊN dấu thanh tiếng Việt** (ma ≠ má ≠ mà ≠ mả ≠ mã ≠ mạ)
- Không dùng unidecode/strip_accents

### 5.4 Kết quả `compute_dataset_metrics()` trả về

```python
{
    "anls":     float,   # 0–1
    "em":       float,   # 0–1
    "f1":       float,   # 0–1
    "anls_pct": float,   # 0–100  ← metric chính để report
    "em_pct":   float,
    "f1_pct":   float,
    "n_samples": int,
    "per_sample": [ {"idx", "pred", "gts", "anls", "em", "f1"}, ... ]
}
```

---

## 6. Logging & Visualization (BẮT BUỘC)

### 6.1 JSON Logger (`src/utils/logger.py`)

**Mọi sự kiện đều phải được log** — thiết kế append-only:

```
logs/
└── {model}_{stage}_{timestamp}.json
    {
        "run_id": "...",
        "config": { tất cả training args },
        "events": [
            {"type": "model_info", "total_params": ..., ...},
            {"type": "train_step", "stage": 1, "global_step": 10,
             "epoch": 0.1, "loss": 2.34, "learning_rate": 1e-3, ...},
            {"type": "eval", "stage": 3, "split": "test",
             "anls": 0.72, "em": 0.41, "f1": 0.68, ...},
            {"type": "stage_end", "duration_sec": 3600, ...}
        ]
    }
```

**Các class/method log:**
- `ViLeoMiniLogger.log_train_step()` — mỗi logging_steps bước
- `ViLeoMiniLogger.log_eval()` — sau mỗi evaluation (bao gồm per_sample)
- `ViLeoMiniLogger.log_stage_end()` — khi stage kết thúc
- `LeoMiniLoggingCallback` — HuggingFace TrainerCallback tự động hook vào Trainer

**Mỗi run tạo 1 file JSON duy nhất** — không split. Dùng `get_events("train_step")` để query.

### 6.2 Chart Export (`src/utils/visualizer.py`)

Tất cả charts được lưu dưới `output_dir/plots/` dạng PNG (dpi=150):

| File | Nội dung |
|---|---|
| `training_loss_stage{N}.png` | Loss curve theo step, mỗi stage 1 file |
| `balance_loss_stage3.png` | Total loss vs balance loss (dual-axis) |
| `eval_metrics_history.png` | ANLS/EM/F1 (%) qua các step eval |
| `learning_rate_schedule.png` | LR schedule các stage trên cùng trục |
| `anls_distribution_stage{N}_{split}.png` | Histogram ANLS per-sample |

**Trigger**: `plot_training_history(log_path, output_dir)` — generate tất cả từ 1 log JSON.
Được tự động gọi ở cuối `vi_evaluator.py` (xem §8.2).

---

## 7. Checkpoint Save/Load Protocol (QUAN TRỌNG)

### 7.1 Những gì `train()` lưu sau mỗi stage

**Stage 1 và 2** — lưu 2 artifacts tại `output_dir/stage{N}/`:
```
checkpoints/qwen2_5_3b_vi/stage{N}/
├── llm_checkpoint/          ← model.llm.save_pretrained()  [HF format]
└── projector_weights.pt     ← model.save_projector_weights()
```

**Stage 3** — lưu 1 artifact:
```
checkpoints/qwen2_5_3b_vi/stage3/
└── stage3_adapter_weights.pt  ← model.save_stage3_weights()
    (chứa: projector + CoTR + MMoE-LLM LoRA, không có LLM backbone)
```

> **KHÔNG dùng `trainer.save_model()`** cho stage handoff.

### 7.2 Auto-derive checkpoint paths khi có `--model_config`

```
Stage 1: llm_path = base_model_path (Qwen/Qwen2.5-3B-Instruct)
Stage 2: llm_path = output_dir/stage1/llm_checkpoint/
         projector_path = output_dir/stage1/projector_weights.pt
Stage 3: llm_path = output_dir/stage2/llm_checkpoint/
         projector_path = output_dir/stage2/projector_weights.pt
```

### 7.3 Khi prev stage checkpoint không tồn tại

→ `sys.exit(1)` với error message rõ ràng. **Không silent fallback**.

---

## 8. Cấu trúc Source Code

```
src/
├── models/
│   ├── cotr.py           CoTR: Eq.2–7. Class CoTR(nn.Module)
│   ├── mmoe_llm.py       MMoE-LLM: LoRALayer, MMoERouter, MMoELinear,
│   │                     apply_mmoe_to_model(), collect_balance_loss()
│   ├── vision_experts.py CLIPExpert, Pix2StructExpert, MMoEVision,
│   │                     build_mmoe_vision(experts=["clip","pix2struct"])
│   │                     (EVA02Expert, ConvNeXtExpert vẫn có nhưng không dùng)
│   ├── projector.py      VisualProjector (2-layer MLP)
│   └── leo_mini.py       LeoMini: full pipeline, Qwen2.5 tokenizer aware
├── data/
│   ├── dataset.py        EAGLEDataset, LLaVADataset (KHÔNG dùng cho Vi variant)
│   ├── collator.py       LeoMiniCollator: padding + pixel_values stacking
│   └── vitextvqa_dataset.py   ← CHÍNH CHO VI VARIANT
│       ├── VietnameseMultimodalDataset  (base class)
│       ├── KTVICDataset                 (Stage 1)
│       ├── OpenViVQADataset             (Stage 2, JSON loader đặc biệt)
│       └── ViTextVQADataset             (Stage 3)
├── train/
│   └── trainer.py        LeoMiniTrainer, train(), stage-dispatch logic
├── eval/
│   ├── metrics.py        anls_score(), exact_match_score(), f1_score(),
│   │                     compute_dataset_metrics(), normalize_vi()
│   ├── vi_evaluator.py   ViTextVQAEvaluator, save_eval_result(),
│   │                     load_eval_history(), EvalResult dataclass
│   ├── evaluator.py      Original English evaluator (ít dùng cho Vi variant)
│   └── lmms_adapter.py   @register_model("leomini") cho lmms-eval
└── utils/
    ├── logger.py         ViLeoMiniLogger, LeoMiniLoggingCallback, make_run_id()
    └── visualizer.py     plot_training_history(), plot_anls_distribution(),
                          plot_training_loss(), plot_balance_loss(),
                          plot_eval_history(), plot_lr_schedule()
```

---

## 9. Config System

### 9.1 Stage configs cho Vi-LEO-MINI

```
configs/
├── config_vi_stage1.yaml   # Stage 1: KTVIC, projector-only, lr=1e-3
├── config_vi_stage2.yaml   # Stage 2: OpenViVQA, full SFT, adamw_bnb_8bit
└── config_vi_stage3.yaml   # Stage 3: ViTextVQA, CoTR+LoRA, n_visual=128
```

### 9.2 Model config

```
configs/models/qwen2_5_3b_vi.yaml
```

```yaml
base_model_path: "Qwen/Qwen2.5-3B-Instruct"
output_dir:      "checkpoints/qwen2_5_3b_vi"
vision_experts:  [clip, pix2struct]   # 2-expert subset
n_visual:        128
vi_train_path:   "train"
vi_val_path:     "validation"
log_dir:         "logs/qwen2_5_3b_vi"
optim:           "adamw_bnb_8bit"
```

**Quy tắc tuyệt đối**:
- Dùng `base_model_path` (không phải `llm_path`) để không ghi đè checkpoint path
- `vi_train_path: "train"` kích hoạt Vi dataset path trong trainer

### 9.3 Merge priority (trainer.py `__main__`)

1. `llm_path` explicit trong model_cfg → dùng ngay
2. `base_model_path` → auto-derive theo stage
3. Không có gì → giữ stage config's `llm_path`

---

## 10. Training Scripts

```bash
# Stage 1 — Projector Warmup trên KTVIC
bash scripts/run_vi_stage1.sh configs/models/qwen2_5_3b_vi.yaml

# Stage 2 — Full SFT trên OpenViVQA
bash scripts/run_vi_stage2.sh configs/models/qwen2_5_3b_vi.yaml

# Stage 3 — CoTR + MMoE-LLM trên ViTextVQA
bash scripts/run_vi_stage3.sh configs/models/qwen2_5_3b_vi.yaml

# Multi-GPU
NUM_GPUS=2 bash scripts/run_vi_stage1.sh configs/models/qwen2_5_3b_vi.yaml
```

---

## 11. Evaluation

### 11.1 Single-stage evaluation trên ViTextVQA

```bash
python -m src.eval.vi_evaluator \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --stage3_weights checkpoints/qwen2_5_3b_vi/stage3/stage3_adapter_weights.pt \
    --split test \
    --output_dir results/vi_leomini/ \
    --log_dir logs/ \
    --vision_experts clip pix2struct \
    --n_visual 128
```

Sau khi chạy tự động generate tất cả charts vào `results/vi_leomini/plots/`.

### 11.2 Output artifacts sau evaluation

```
results/vi_leomini/
├── eval_stage3_test_{timestamp}.json   ← per-sample results + aggregated metrics
└── plots/
    ├── training_loss_stage1.png
    ├── training_loss_stage2.png
    ├── training_loss_stage3.png
    ├── balance_loss_stage3.png
    ├── eval_metrics_history.png
    ├── learning_rate_schedule.png
    └── anls_distribution_stage3_test.png
```

### 11.3 Report format

| Metric | Format |
|---|---|
| **ANLS** | `anls_pct` (0–100%) — **metric chính** |
| EM | `em_pct` (0–100%) |
| F1 | `f1_pct` (0–100%) |

---

## 12. Hardware & VRAM

| Stage | VRAM | Optimizer |
|---|---|---|
| Stage 1 (projector only) | ~12 GB | adamw_torch |
| Stage 2 (full SFT) | ~30 GB | **adamw_bnb_8bit** |
| Stage 3 (CoTR + LoRA) | ~15 GB | adamw_bnb_8bit |

Thử nghiệm trên 1× GPU 48 GB (RTX A6000 hoặc H100-PCIe).

---

## 13. Key Classes & Functions

### `LeoMini` (leo_mini.py)

- `set_stage(stage)` — freeze/unfreeze theo stage
- `forward(input_ids, attention_mask, pixel_values, pix2struct_inputs, labels)` — thêm L_balance ở Stage 3
- `generate(...)` — inference với visual tokens merged
- `from_pretrained(llm_path, vision_experts=["clip","pix2struct"], n_visual=128, ...)` — builder
- `save_stage3_weights(path)` — lưu projector + CoTR + MMoE-LLM
- `save_projector_weights(path)` — lưu chỉ projector

### `ViTextVQAEvaluator` (vi_evaluator.py)

- `evaluate(split, hf_dataset_name, limit, global_step)` → `EvalResult`
- `EvalResult.to_dict()` → JSON-serializable dict
- `save_eval_result(result, output_dir)` → lưu JSON, trả về path
- `load_eval_history(output_dir)` → list of all past eval JSONs

### `ViLeoMiniLogger` (logger.py)

- `log_train_step(stage, global_step, epoch, loss, learning_rate, ...)` 
- `log_eval(stage, global_step, split, anls, em, f1, n_samples, per_sample)`
- `log_stage_end(stage, duration_sec, checkpoint_path)`
- `LeoMiniLoggingCallback` — attach to HuggingFace Trainer via `trainer.add_callback()`

### `CoTR` (cotr.py)

- `forward(visual_tokens: List[Tensor], text_tokens: Tensor)` → `(B, 128, 3072)` cho Vi-LEO-MINI
- Mỗi expert có learnable query `Q_i ∈ R^{128 × d_i}`
- Projection: `proj_Q[i]`, `proj_I[i]` (per-expert), `proj_T` (shared)

---

## 14. Invariants (KHÔNG ĐƯỢC PHÁ VỠ)

1. **`base_model_path` không phải `llm_path`** trong model configs — tránh ghi đè checkpoint path.

2. **`model.llm.save_pretrained()` cho stage handoff** — `trainer.save_model()` lưu full LeoMini state dict, không loadable bằng `AutoModelForCausalLM.from_pretrained()`.

3. **`sys.exit(1)` khi missing checkpoint** — không fallback.

4. **`vision_experts=["clip", "pix2struct"]` luôn phải được set** cho Vi-LEO-MINI — nếu để mặc định (4 experts) sẽ load thêm EVA-02 và ConvNeXt không cần thiết, gây OOM và sai d^V.

5. **`n_visual=128` phải nhất quán** giữa config_vi_stage{1,2,3}.yaml, model config, và evaluator CLI args — mismatch → shape error trong CoTR.

6. **Vietnamese normalization GIỮ dấu thanh** — không dùng unidecode. Xem `normalize_vi()` trong `metrics.py`.

7. **Qwen2.5 tokenizer cần `use_fast=True`** — đã có trong `leo_mini.py`. Không đổi lại False.

8. **OpenViVQA JSON loader đặc biệt** — dataset có cấu trúc `{"images": {...}, "annotations": {...}}` với file ảnh trong zip riêng. `_load_with_fallback()` trong `vitextvqa_dataset.py` xử lý case này. Không thay bằng `load_dataset()` thông thường.

9. **ANLS threshold = 0.5** — chuẩn TextVQA, không thay đổi khi so sánh với các paper khác.

10. **Log PHẢI được ghi trước khi crash** — `ViLeoMiniLogger._flush()` gọi sau mỗi event. JSON file luôn valid ngay cả khi training bị interrupt.

---

## 15. Checkpoint Directory Structure

```
checkpoints/qwen2_5_3b_vi/
├── stage1/
│   ├── llm_checkpoint/          ← llm_path cho Stage 2
│   ├── projector_weights.pt     ← projector_path cho Stage 2
│   └── checkpoint-{N}/          ← HF Trainer intermediate
├── stage2/
│   ├── llm_checkpoint/          ← llm_path cho Stage 3
│   ├── projector_weights.pt     ← projector_path cho Stage 3
│   └── checkpoint-{N}/
└── stage3/
    └── stage3_adapter_weights.pt  ← dùng cho evaluation

logs/qwen2_5_3b_vi/
├── Qwen2_5_3B_Instruct_stage1_{timestamp}.json
├── Qwen2_5_3B_Instruct_stage2_{timestamp}.json
└── Qwen2_5_3B_Instruct_stage3_{timestamp}.json

results/vi_leomini/
├── eval_stage3_test_{timestamp}.json
└── plots/
    └── *.png
```

---

## 16. Blockers đã được fix

| ID | Mô tả | Fix |
|---|---|---|
| B1 | `cfg.update()` ghi đè llm_path | `base_model_path` key + priority-based merge |
| B2 | Output dir collision | `output_dir` riêng trong model config |
| B3 | `transformers==4.40.0` | `transformers>=4.43.0` |
| B4 | `**kwargs` bị drop | Pass `**kwargs` tới `AutoModelForCausalLM.from_pretrained()` |
| B5 | KTVIC download path sai | Fix `hf_dataset_name` Stage 1 → `ai-enthusiasm-community/KTVIC` |
| B6 | OpenViVQA download path sai | Fix `hf_dataset_name` Stage 2 → `uitnlp/OpenViVQA-dataset` |
| B7 | OpenViVQA JSON parse fail | Custom `_load_with_fallback()` cho cấu trúc `{images, annotations}` + zip |
| B8 | Stage 3: "1 sample" + missing `pixel_values` | `minhquan6203/ViTextVQA` dùng JSON riêng (ViTextVQA_{split}.json) + ảnh trong ViTextVQA_images.zip. Custom loader đã được thêm vào `_load_with_fallback`. Split "validation" → "dev" trong HF repo. |
| NB1 | Stage 1→2 mất projector | Auto-derive `projector_path` |
| NB2 | `trainer.save_model()` không loadable | Dùng `model.llm.save_pretrained()` |
| NB3 | Silent fallback khi missing ckpt | `sys.exit(1)` |

---

## 17. Quick Reference

```
run_vi_stage{N}.sh
  └─ python -m src.train.trainer --config config_vi_stage{N}.yaml --model_config qwen2_5_3b_vi.yaml
       └─ __main__: smart merge → LeoMiniTrainingArgs
            └─ train(args)
                 ├─ ViLeoMiniLogger(log_path, run_id, config)
                 ├─ LeoMini.from_pretrained(
                 │       llm_path,
                 │       vision_experts=["clip","pix2struct"],
                 │       n_visual=128
                 │   )
                 │    ├─ Qwen2.5-3B-Instruct backbone
                 │    ├─ CLIPExpert + Pix2StructExpert  (d^V=3072)
                 │    ├─ VisualProjector(3072, 2048)
                 │    └─ [Stage 3] CoTR(expert_dims=[1024,2048], n_visual=128)
                 │                + apply_mmoe_to_model()
                 ├─ model.set_stage(N)
                 ├─ Dataset dispatch:
                 │    Stage 1 → KTVICDataset("ai-enthusiasm-community/KTVIC")
                 │    Stage 2 → OpenViVQADataset("uitnlp/OpenViVQA-dataset")
                 │    Stage 3 → ViTextVQADataset("minhquan6203/ViTextVQA")
                 ├─ LeoMiniTrainer + LeoMiniLoggingCallback(vi_logger)
                 └─ save: llm_checkpoint/ + projector_weights.pt (Stage 1/2)
                          stage3_adapter_weights.pt              (Stage 3)

python -m src.eval.vi_evaluator
  └─ ViTextVQAEvaluator.evaluate(split="test")
       ├─ compute_dataset_metrics(predictions, ground_truths)
       │    → ANLS_pct / EM_pct / F1_pct
       ├─ vi_logger.log_eval(...)      → append to JSON
       ├─ save_eval_result(...)        → eval_{timestamp}.json
       └─ plot_training_history(...)   → plots/*.png
```
