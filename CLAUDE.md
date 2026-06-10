# CLAUDE.md — LEO-MINI Project Context

> Đọc file này trước mỗi session làm việc với repo này.
> Cập nhật khi có thay đổi kiến trúc, fix quan trọng, hoặc quyết định thiết kế mới.

---

## 1. Mục tiêu dự án

Tái hiện thực nghiệm bài báo **LEO-MINI** (EMNLP 2025):

> "LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token
> Reduction and Mixture of Multi-Modal Experts"
> — Wang et al., University of Waterloo

Mục tiêu cụ thể: train + eval LEO-MINI trên **1 GPU 48 GB** thay vì 8× A6000 như paper,
dùng 3 LLM backbone nhỏ hơn (Llama-3.2-1B/3B, Phi-3.5-mini) thay cho Llama3-8B.

Paper PDF: `docs/leomini.pdf`

---

## 2. Hai đóng góp cốt lõi của bài báo

### 2.1 CoTR — Conditional Token Reduction (Section 3.2, Eq. 2–7)

Giảm mỗi expert từ 576 tokens → 64 tokens bằng weighted aggregation với 4 loại attention:

```
Eq.2  s_QUERY  = Q̄_i · Ī_i^T          ∈ R^{N^V × N_i}   (query-visual)
Eq.3  s_SELF   = 1 · Ī_i · Ī_i^T       ∈ R^{1 × N_i}     (self-similarity)
Eq.4  s_CROSS  = Σ_{j≠i} 1·Ī_j·Ī_i^T  ∈ R^{1 × N_i}     (cross-expert)
Eq.5  s_TEXT   = 1 · T̂ · Ī_i^T         ∈ R^{1 × N_i}     (text-visual)
Eq.6  α_i = softmax((Σ scores) / √d_i^V)                  (scale bởi DIM GỐC, không phải d_proj)
Eq.7  Ī_i = α_i · I_i  →  cat([Ī_1,...,Ī_m])             (aggregation trong original space)
```

**Quan trọng**: "1" là ones-vector → tổng token positions (sum, không phải mean).
Scale factor dùng `expert_dims[i]` (feature dim gốc), không phải `d_proj`.

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

## 3. Kiến trúc tổng thể

```
Image(s) + Text
    │
    ├─ MMoE-Vision (4 experts, tất cả frozen trong Stage 3)
    │   ├─ CLIP ViT-L/14-336     → (B, 576, 1024)
    │   ├─ EVA-02 CLIP-L-14-336  → (B, 576, 1024)
    │   ├─ ConvNeXt-Large-D      → (B, 576,  768)
    │   └─ Pix2Struct-Large      → (B, 576, 2048)
    │   Tổng d^V = 4864
    │
    ├─ CoTR [Stage 3 only]  → (B, 64, 4864)
    │   Stage 1+2: simple cat, không qua CoTR → (B, 576, 4864)
    │
    ├─ VisualProjector  2-layer MLP (d^V → 4·d_LLM → d_LLM)
    │   → (B, N^V, d_LLM)  (N^V = 64 Stage 3, 576 Stage 1+2)
    │
    └─ LLM + MMoE-LLM [Stage 3 only]
        concat([visual_tokens, text_tokens]) → autoregressive generation
        IMAGE_TOKEN_INDEX = -200 (sentinel trong input_ids)
```

### Hyperparameters từ paper (Section 4.1)

| Param | Value |
|---|---|
| N^V (visual tokens sau CoTR) | 64 |
| d_proj_cotr (projection dim) | 256 |
| lora_rank | 16 |
| num_special (E, số LoRA experts) | 3 |
| balance_loss_lambda | 0.05 |

---

## 4. 3-Stage Training (Table 6, paper)

| Stage | Trainable | Frozen | Data |
|---|---|---|---|
| 1 Projector Warmup | VisualProjector | LLM, Vision Experts | EAGLE alignment |
| 2 Full SFT | Tất cả | — | EAGLE SFT |
| 3 Token Reduction | CoTR + MMoE-LLM LoRAs + Routers + Projector | LLM backbone, Vision Experts | LLaVA-v1.5 665K |

**Lưu ý Stage 3**: Projector cũng trainable trong Stage 3 (paper Table 6), không chỉ CoTR và MMoE-LLM.

---

## 5. Checkpoint Save/Load Protocol (QUAN TRỌNG)

Đây là phần dễ sai nhất. Phải hiểu rõ trước khi sửa `trainer.py`.

### 5.1 Những gì `train()` lưu sau mỗi stage

**Stage 1 và 2** — lưu 2 artifacts tại `output_dir/stage{N}/`:
```
output_dir/stage{N}/
├── llm_checkpoint/          ← model.llm.save_pretrained()  [HF format]
│   ├── config.json
│   ├── model.safetensors (hoặc pytorch_model.bin)
│   └── tokenizer.json, ...
└── projector_weights.pt     ← model.save_projector_weights()
```

**Stage 3** — lưu 1 artifact:
```
output_dir/stage3/
└── stage3_adapter_weights.pt  ← model.save_stage3_weights()
    (chứa: projector + CoTR + MMoE-LLM LoRA weights, không có LLM backbone)
```

> **KHÔNG dùng `trainer.save_model()`** cho stage handoff — nó lưu full LeoMini
> nn.Module state dict (prefix `llm.`, `vision.`, `projector.`) mà
> `AutoModelForCausalLM.from_pretrained()` không đọc được.

### 5.2 Những gì `__main__` auto-derive khi có `--model_config`

```
Stage 1: llm_path = base_model_path (HuggingFace ID)
Stage 2: llm_path = output_dir/stage1/llm_checkpoint/
         projector_path = output_dir/stage1/projector_weights.pt
Stage 3: llm_path = output_dir/stage2/llm_checkpoint/
         projector_path = output_dir/stage2/projector_weights.pt
```

### 5.3 Khi prev stage checkpoint không tồn tại

→ `sys.exit(1)` với error message rõ ràng. **Không silent fallback**.

Override: thêm `llm_path: "NVEagle/Eagle-X4-8B-Plus"` vào model config để skip stages.

---

## 6. Config System

### 6.1 Stage configs (base configs)

```
configs/
├── config_stage1.yaml   # Stage 1 defaults (Llama3-8B, multi-GPU)
├── config_stage2.yaml   # Stage 2 defaults
└── config_stage3.yaml   # Stage 3 defaults (NVEagle/Eagle-X4-8B-Plus)
```

`llm_path` trong stage 2/3 là fallback cho trường hợp không dùng `--model_config`.

### 6.2 Model configs (overrides)

```
configs/models/
├── llama3_2_1b.yaml   # base_model_path + output_dir + batch/optim settings
├── llama3_2_3b.yaml
└── phi3_5_mini.yaml
```

**Quy tắc tuyệt đối**:
- Model configs dùng `base_model_path` (không phải `llm_path`) để không ghi đè checkpoint path
- Model configs có `output_dir` riêng để cách ly checkpoint giữa các models
- Model configs có thể có `llm_path` explicit để skip stages (ví dụ: EAGLE shortcut)

### 6.3 Merge logic (trainer.py `__main__`)

Priority:
1. `llm_path` explicit trong model_cfg → dùng ngay (skip-stage scenario)
2. `base_model_path` → auto-derive theo stage (xem 5.2)
3. Không có gì → giữ nguyên stage config's `llm_path`

---

## 7. Supported Models & Hardware

| Model | VRAM Stage 1 | VRAM Stage 2 | VRAM Stage 3 | Optimizer Stage 2 |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | ~8 GB | ~14 GB | ~8 GB | adamw_torch |
| Llama-3.2-3B-Instruct | ~12 GB | ~27 GB | ~14 GB | **adamw_bnb_8bit** |
| Phi-3.5-mini-instruct | ~13 GB | ~30 GB | ~15 GB | **adamw_bnb_8bit** |
| Meta-Llama-3-8B-Instruct (paper) | — | ~65 GB | ~32 GB | DeepSpeed Zero2 |

**Phi-3.5-mini notes**:
- Cần `transformers>=4.43.0` (Phi3ForCausalLM chưa có trong 4.40)
- MLP có `gate_up_proj` (fused) thay vì `gate_proj`+`up_proj`, nhưng `down_proj` vẫn tồn tại → `apply_mmoe_to_model()` hoạt động bình thường
- `use_fast=True` bắt buộc (Tiktoken tokenizer)

---

## 8. Cấu trúc Source Code

```
src/
├── models/
│   ├── cotr.py           CoTR: Eq.2–7. Class CoTR(nn.Module)
│   ├── mmoe_llm.py       MMoE-LLM: LoRALayer, MMoERouter, MMoELinear,
│   │                     apply_mmoe_to_model(), collect_balance_loss()
│   ├── vision_experts.py CLIPExpert, EVA02Expert, ConvNeXtExpert,
│   │                     Pix2StructExpert, MMoEVision, build_mmoe_vision()
│   ├── projector.py      VisualProjector (2-layer MLP)
│   └── leo_mini.py       LeoMini: full pipeline, set_stage(), from_pretrained(),
│                         save_stage3_weights(), save_projector_weights()
├── data/
│   ├── dataset.py        EAGLEDataset, LLaVADataset (cùng MultimodalDataset)
│   └── collator.py       LeoMiniCollator: padding + pixel_values stacking
├── train/
│   └── trainer.py        LeoMiniTrainingArgs, LeoMiniTrainer, train(),
│                         __main__ với smart merge logic
└── eval/
    ├── evaluator.py      LeoMiniEvaluator, compare_models(), run_token_ablation()
    └── lmms_adapter.py   @register_model("leomini") cho lmms-eval framework
```

---

## 9. Key Classes & Functions

### `LeoMini` (leo_mini.py)

- `set_stage(stage)` — freeze/unfreeze theo stage (xem Table 6)
- `forward(input_ids, attention_mask, pixel_values, pix2struct_inputs, labels)` — multimodal forward, thêm L_balance ở Stage 3
- `generate(...)` — inference với visual tokens merged
- `from_pretrained(llm_path, projector_path, stage3_weights, ...)` — builder, hỗ trợ 4-bit quant
- `save_stage3_weights(path)` — lưu projector + CoTR + MMoE-LLM (không lưu backbone)
- `save_projector_weights(path)` — lưu chỉ projector

### `MMoELinear` (mmoe_llm.py)

- Drop-in replacement cho `nn.Linear` trong LLM MLP
- `forward(x)` — trả về `f_ORI(x) + f_GEN(x) + f_top1(x)` với straight-through estimator
- `get_balance_loss(lambda_balance)` → scalar loss
- `reset_balance_stats()` — gọi sau mỗi bước train

### `CoTR` (cotr.py)

- `forward(visual_tokens: List[Tensor], text_tokens: Tensor)` → `(B, N^V, Σd_i)`
- Mỗi expert có learnable query `Q_i ∈ R^{N^V × d_i}`
- Projection: `proj_Q[i]`, `proj_I[i]` (per-expert), `proj_T` (shared)

### `ContextBuffer` (mmoe_llm.py)

- Shared buffer đặt trước khi LLM forward để tất cả `MMoELinear` layers có thể đọc `(Ī, T)`
- `set(visual, text)` → gọi trong `LeoMini.forward()` trước khi gọi `self.llm(...)`
- `clear()` → gọi sau khi LLM forward xong

---

## 10. Evaluation System

### Single model
```bash
python -m src.eval.evaluator \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --model_name "LEO-MINI-3B" \
  --tasks pope,scienceqa,textvqa,mme,mmmu \
  --output_dir results/3b
```

`model_path` = base LLM path (không phải stage3_adapter_weights.pt).
Stage3 weights được pass riêng qua `LeoMini.from_pretrained(stage3_weights=...)`.

### Multi-model comparison
```bash
BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
TASKS="pope,scienceqa,textvqa" LIMIT=200 LOAD_4BIT=1 \
bash scripts/run_benchmark_compare.sh
```

`--model_specs` format: `"Name:base_path"` hoặc `"Name:base_path:stage3_weights_path"`.

### lmms-eval integration

`scripts/run_eval_leomini.py` là wrapper để register `@register_model("leomini")` trước khi
lmms-eval CLI chạy. `evaluator.py` gọi script này qua subprocess, không gọi `lmms-eval` trực tiếp.

---

## 11. Training Scripts

```bash
# Stage 1 (chọn model config)
bash scripts/run_stage1.sh configs/models/llama3_2_3b.yaml

# Stage 2 (sẽ fail nếu Stage 1 chưa chạy — sys.exit(1))
bash scripts/run_stage2.sh configs/models/llama3_2_3b.yaml

# Stage 3
bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml

# Multi-GPU
NUM_GPUS=4 bash scripts/run_stage1.sh configs/models/llama3_2_3b.yaml
```

---

## 12. Data Format

EAGLE và LLaVA dùng cùng JSON format:
```json
[
  {
    "id": "...",
    "image": "relative/path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt",   "value": "..."}
    ]
  }
]
```

`image_dir` config = root thư mục chứa ảnh. `data_path` = đường dẫn tới file JSON.

---

## 13. Decisions & Invariants (KHÔNG ĐƯỢC PHÁ VỠ)

1. **`base_model_path` trong model configs, không phải `llm_path`** — Nếu đổi tên, trainer sẽ ghi đè stage checkpoint path, phá stage handoff.

2. **`model.llm.save_pretrained()` cho stage handoff** — `trainer.save_model()` không tương thích với `AutoModelForCausalLM.from_pretrained()`. Đây là quyết định thiết kế quan trọng nhất của trainer.

3. **`sys.exit(1)` khi missing checkpoint** — Không fallback về base model. Phải train stages theo thứ tự hoặc explicit override bằng `llm_path` trong model config.

4. **`projector_path` phải được set cho Stage 2 và 3** — Stage 2 cần warm projector từ Stage 1. Nếu thiếu → random init → mất Stage 1 warmup. Auto-derived khi `--model_config` được dùng.

5. **`AutoModelForCausalLM` thay vì `LlamaForCausalLM`** — Hỗ trợ Phi-3.5-mini và tất cả Llama variants.

6. **`use_fast=True` cho tokenizer** — Bắt buộc cho Phi-3.5-mini (Tiktoken). Không đổi lại `False`.

7. **CoTR scale factor = `sqrt(expert_dims[i])`** — Dùng dim gốc của expert, không phải `d_proj`. Đây là thực tế trong paper (Eq. 6).

8. **Stage 3 Projector là trainable** — Paper Table 6 ghi rõ projector được train ở Stage 3, không chỉ CoTR và MMoE-LLM.

9. **`**kwargs` pass-through trong `from_pretrained()`** — Cần cho `trust_remote_code`, `attn_implementation` với các model đặc biệt.

10. **`transformers>=4.43.0`** — Phi-3.5-mini cần. Không downgrade về 4.40.

---

## 14. Blockers đã được fix

| Blocker | Mô tả | Fix |
|---|---|---|
| B1 `cfg.update()` ghi đè llm_path | Stage 2/3 load lại base model | `base_model_path` key + priority-based merge |
| B2 Output dir collision | 3 models ghi chung `checkpoints/` | `output_dir` riêng trong mỗi model config |
| B3 Benchmark compare sai semantic | `AutoModelForCausalLM(stage3_dir)` fail | `--model_specs "Name:base:stage3_path"` format |
| B4 `transformers==4.40.0` | Phi-3.5-mini ImportError | `transformers>=4.43.0` |
| B4 `**kwargs` bị drop | `trust_remote_code` không pass được | Pass `**kwargs` tới `AutoModelForCausalLM.from_pretrained()` |
| NB1 Stage 1→2 mất projector | Stage 2 random init projector | Auto-derive `projector_path` cho Stage 2 |
| NB2+3 `trainer.save_model()` không loadable | Stage handoff fail | Dùng `model.llm.save_pretrained()` |
| NB4 Warning+fallback silently sai | User không biết stage bị skip | `sys.exit(1)` khi missing checkpoint |

---

## 15. Checkpoint Directory Structure (after full training)

```
checkpoints/llama3_2_3b/
├── stage1/
│   ├── llm_checkpoint/          ← dùng làm llm_path cho Stage 2
│   ├── projector_weights.pt     ← dùng làm projector_path cho Stage 2
│   └── checkpoint-500/          ← HF Trainer intermediate (cho resume)
├── stage2/
│   ├── llm_checkpoint/          ← dùng làm llm_path cho Stage 3
│   ├── projector_weights.pt     ← dùng làm projector_path cho Stage 3
│   └── checkpoint-500/
└── stage3/
    └── stage3_adapter_weights.pt  ← dùng cho evaluation & inference
```

---

## 16. Quick Reference: Quan hệ giữa các file

```
run_stageN.sh
  └─ python -m src.train.trainer --config config_stageN.yaml --model_config model.yaml
       └─ __main__: smart merge → LeoMiniTrainingArgs
            └─ train(args)
                 ├─ LeoMini.from_pretrained(llm_path, projector_path, ...)
                 │    ├─ AutoModelForCausalLM.from_pretrained(llm_path, **kwargs)
                 │    ├─ build_mmoe_vision()  [CLIP + EVA + ConvNeXt + Pix2Struct]
                 │    ├─ VisualProjector(d_visual, d_llm)
                 │    ├─ [Stage 3] CoTR(expert_dims, d_text, d_proj, n_visual)
                 │    └─ [Stage 3] apply_mmoe_to_model(llm, ...)
                 ├─ model.set_stage(N)
                 ├─ LeoMiniTrainer.train(...)
                 └─ save: llm_checkpoint/ + projector_weights.pt (Stage 1/2)
                          stage3_adapter_weights.pt              (Stage 3)

run_benchmark_compare.sh
  └─ python -m src.eval.evaluator --compare --model_specs "Name:base:ckpt,..."
       └─ compare_models()
            └─ LeoMiniEvaluator.run(tasks)
                 └─ subprocess: scripts/run_eval_leomini.py --model leomini --model_args pretrained=...,stage3_weights=...
                      └─ import src.eval.lmms_adapter  [registers @register_model("leomini")]
                           └─ LeoMiniAdapter → LeoMini.from_pretrained(...)
```
