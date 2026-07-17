# 📄 LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

## 📌 TL;DR
LeWorldModel (LeWM) is the first JEPA-based world model that trains stably end-to-end from raw pixels using only **two loss terms**: an MSE next-embedding prediction loss and a Sketched Isotropic Gaussian Regularizer (SIGReg) that prevents representation collapse by enforcing a Gaussian latent distribution. This eliminates the need for stop-gradient, EMA encoders, or pre-trained backbones, reducing tunable loss hyperparameters from six (PLDM) to one. The codebase implements the full model architecture, training loop, and MPC-based latent planning pipeline faithfully to the paper.

---

## 🔬 Research Overview & Core Problem

* **The Problem:** Existing JEPA world models suffer from *representation collapse* — the encoder maps all observations to a single constant embedding to trivially minimize the prediction loss. Prior fixes (EMA target encoders, stop-gradient, VICReg with 6+ loss terms, frozen pre-trained encoders) introduce training instability, hyperparameter bloat, or limit generality by tying the model to pre-trained representations.
* **Key Contributions:**
  * First end-to-end JEPA stable from raw pixels with a two-term objective (prediction MSE + SIGReg).
  * Reduces tunable loss hyperparameters from 6 → 1 (`λ`, the SIGReg regularization weight).
  * Achieves up to **48× faster** planning vs. DINO-WM (foundation-model-based competitor) with ~15M parameters, trainable on a single GPU in a few hours.
  * Demonstrates that latent space encodes meaningful physical structure (agent/block positions, angles) via linear and MLP probing.
  * Introduces Violation-of-Expectation (VoE) evaluation: LeWM assigns significantly higher surprise to physically implausible events (object teleportation) versus visual perturbations (color changes).
* **Key Results:**
  * **Push-T:** 96% success rate (vs. PLDM 78%, DINO-WM 74%).
  * **Reacher:** 86% success rate (vs. PLDM 78%, DINO-WM 79%).
  * **OGBench-Cube:** 74% success rate (slightly below DINO-WM's 86%, attributed to 3D visual complexity).
  * **Two-Room:** 87% (below PLDM 97%, attributed to low-dimensionality mismatch with high-dimensional Gaussian prior).
  * Planning time: **0.98s** (LeWM) vs. **47s** (DINO-WM) per full planning episode.

---

## 🏗️ Architecture Deep-Dive

* **High-Level Structure:**
  ```
  Observation (pixels) → Encoder (ViT-Tiny) → [CLS] token → Projector (MLP + BN) → Latent z_t
  (z_t, a_t) → Predictor (Causal Transformer + AdaLN) → Pred Projector (MLP + BN) → ẑ_{t+1}
  ```
  Training:  ``` MSE(ẑ_{t+1}, z_{t+1}) + λ · SIGReg(Z)  ```
  
  Inference:   ```MPC via Cross-Entropy Method (CEM) in latent space.  ```

* **Novel Components:**
  * **Encoder:** Vision Transformer (ViT-Tiny), ~5M parameters. Patch size 14, 12 layers, 3 attention heads, hidden dim 192. The `[CLS]` token from the last layer is extracted and passed through a **Projector** (1-layer MLP + BatchNorm1d). BatchNorm replaces ViT's final LayerNorm to allow SIGReg to operate effectively — LayerNorm blocks the Gaussian shaping signal.
  * **Predictor:** Causal Transformer, ~10M parameters. 6 layers, 16 attention heads, 10% dropout. Receives a **history** of N latent embeddings. Actions are injected via **Adaptive Layer Normalization (AdaLN)** at every layer, initialized to zero so action-conditioning grows progressively during training. A **Pred Projector** (same MLP+BN design) post-processes outputs.
  * **Action Encoder (`Embedder`):** Embeds raw action vectors into the predictor's input dimension.
  * **SIGReg:** Projects batch of embeddings onto M=1024 random unit-norm directions in latent space, then applies the **Epps–Pulley univariate normality test statistic** T(·) to each 1D projection. The average over M projections forms the regularizer. By the Cramér–Wold theorem, matching all 1D marginals implies matching the full joint distribution, making the Gaussian prior theoretically grounded.

* **Loss Functions:**

  **Prediction Loss (MSE, teacher-forced):**
  $$L_{\text{pred}} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)$$

  **SIGReg (Sketched Isotropic Gaussian Regularizer):**
  $$\text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)}), \quad h^{(m)} = Z u^{(m)}, \; u^{(m)} \in \mathcal{S}^{d-1}$$

  **Full Objective:**
  $$L_{\text{LeWM}} \triangleq L_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)$$

---

## 💡 Useful Bits & "The Secret Sauce"

* **The "Trick":** Instead of predicting in pixel space (expensive, wastes capacity on irrelevant texture) or using a frozen pre-trained encoder (limits adaptability), LeWM **predicts future embeddings directly** and shapes the entire embedding distribution to be Gaussian via SIGReg. The Gaussian prior isn't arbitrary — it provides *provable* anti-collapse guarantees via the Cramér–Wold theorem. This is fundamentally different from VICReg-style variance/covariance terms, which lack formal collapse guarantees and require careful joint balancing.

* **Removed Heuristics:**
  * ❌ No **stop-gradient** on target representations
  * ❌ No **Exponential Moving Average (EMA)** target encoder
  * ❌ No **pre-trained encoder** (fully from scratch)
  * ❌ No **reward supervision** or task-specific labels
  * ❌ No **image reconstruction** objective
  * All gradients flow through all components — true end-to-end learning.

* **Crucial Hyperparameters:**
  * `λ = 0.1` — regularization weight for SIGReg. The **only** hyperparameter requiring tuning. Can be found efficiently with **bisection search in O(log n)** (vs. O(n⁶) grid search for PLDM's 6 parameters).
  * `M = 1024` — number of random projections in SIGReg. Ablations show performance is **largely insensitive** to M, so it is effectively a fixed constant.
  * **AdaLN zero-init:** The AdaLN parameters in the predictor are initialized to zero — this is load-bearing for stable early training.
  * **BatchNorm over LayerNorm in the Projector:** Critical for SIGReg to function; LayerNorm in ViT's last layer interferes with the Gaussian shaping signal.

---

## 💻 Codebase Mapping

* **Repository:** [https://github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
* **Tech Stack:** Python 3.10, PyTorch, HuggingFace Transformers (ViT backbone), Hydra (config management), WandB (logging), `stable-worldmodel` and `stable-pretraining` external packages for environment management and ViT utilities.
* **Theory-to-Code Map:**

  * **Full JEPA Model (Encoder + Predictor integration):** Implemented in `jepa.py` under `JEPA(nn.Module)`. The `encode()` method runs the ViT encoder + projector to produce `info["emb"]` (the `[CLS]` latent), and `predict()` runs the predictor + pred_proj to produce `ẑ_{t+1}`.

  * **Encoder (ViT-Tiny + Projector):** The ViT backbone is loaded via `stable_pretraining.backbone.utils.vit_hf(...)` (external package, not directly in this repo). The `Embedder` and `MLP` classes in `module.py` implement the projector and action encoder heads. The `JEPA` constructor accepts `projector=MLP(...)` matching the paper's 1-layer MLP + BatchNorm1d design.

  * **Predictor (Causal Transformer + AdaLN):** Implemented in `module.py` under `ARPredictor`. This is the autoregressive causal transformer with AdaLN action conditioning. Uses temporal causal masking so the predictor cannot attend to future embeddings.

  * **Action Encoder:** Implemented in `module.py` under `Embedder`. Maps raw action vectors to the predictor's embedding dimension.

  * **Projector / Pred Projector (MLP + BN):** Implemented in `module.py` under `MLP`, parameterized with `norm_fn=torch.nn.BatchNorm1d`.

  * **SIGReg Loss:** **⚠️ Not directly present in this repository.** SIGReg is provided by the external `stable-pretraining` package. The training logic calling `SIGReg` is invoked via `train.py` and configured through Hydra configs under `config/train/`, but the SIGReg implementation itself lives upstream in `stable-pretraining`.

  * **Training Loop (Alg. 3 from paper):** Implemented in `train.py`, configured via Hydra YAML files in `config/train/` (e.g., `lewm.yaml`, `data/pusht.yaml`). Uses `stable-pretraining` trainer infrastructure.

  * **Latent Planning / MPC (CEM):** Implemented via the `rollout()` and `get_cost()` / `criterion()` methods in `jepa.py`. The full MPC loop (CEM solver, replanning) is handled by `eval.py` and the `stable-worldmodel` package. The cost function is MSE between predicted final latent `ẑ_H` and goal latent `z_g`, matching Eq. (4) in the paper exactly.

  * **Autoregressive Rollout:** `JEPA.rollout()` in `jepa.py` implements the iterative latent rollout: encodes initial context, then at each step truncates to the last `history_size=3` embeddings and appends the new predicted embedding — matching the autoregressive prediction described in Sec. 3.2.

---

## 🚀 Implementation Notes

* **Setup/Execution:**
  ```bash
  # 1. Create environment
  uv venv --python=3.10
  source .venv/bin/activate
  uv pip install stable-worldmodel[train,env]

  # 2. Download data (HuggingFace / HDF5 format)
  tar --zstd -xvf archive.tar.zst
  export STABLEWM_HOME=/path/to/your/storage

  # 3. Configure WandB in config/train/lewm.yaml, then train
  python train.py data=pusht

  # 4. Evaluate with pretrained checkpoint
  python eval.py --config-name=pusht.yaml policy=pusht/lewm
  ```

* **Hardware Constraints:** The paper states ~15M parameters trainable on a **single GPU** in a few hours. No specific GPU model is mandated, but a modern GPU with sufficient VRAM for ViT-Tiny + causal Transformer is required. No multi-GPU setup is described in the paper or README.

* **Code vs. Paper Discrepancies:**
  * **SIGReg not in-repo:** The core anti-collapse loss (SIGReg / Epps-Pulley test statistic) is not implemented in this repository — it lives in the external `stable-pretraining` package. Readers wanting to study or modify SIGReg must inspect that dependency separately.
  * **ViT backbone not in-repo:** The ViT encoder (including the zero-init AdaLN and specific ViT-Tiny config) is loaded via `stable-pretraining`. The paper specifies patch size 14, 12 layers, 3 heads, hidden dim 192, but this config is passed through `config.json` at checkpoint load time rather than being hardcoded here.
  * **Decoder (for visualization) absent:** Fig. 7 / Fig. 10 in the paper decode latents back to pixels via a decoder "trained a posteriori." No decoder implementation is included in this repository.
  * **Physical probing evaluation absent:** The probing experiments from Sec. 5.1 (linear/MLP probes on agent location, block location, block angle) and the t-SNE visualizations are not present in the repo — no probing scripts found.
  * **Violation-of-Expectation (VoE) evaluation absent:** The surprise quantification evaluation from Sec. 5.2 is not implemented here; it likely resides in `stable-worldmodel` scripts.
  * **Baseline implementations absent:** PLDM, DINO-WM, GCBC, GCIVL, GCIQ are referenced in the README as available via the `stable-worldmodel` scripts folder, not this repository.
