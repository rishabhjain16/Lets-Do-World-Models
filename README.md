# Lets-Do-World-Models 🌍 

The goal of this project is to bridge the gap between heavy theoretical math in research papers and practical engineering implementations in code. It serves as a personal knowledge base and a resource for the broader AI research community.

## 🗂️ Paper Index & Code Maps

Below is the index of papers currently broken down in this repository. Each link leads to a comprehensive markdown file containing an architectural deep-dive, the "useful bits" (hyperparameters, tricks), and a direct mapping of the paper's theory to its GitHub codebase.

| Status | Paper / Architecture | Key Focus | Notes Link | ArXiv |
| :---: | :--- | :--- | :--- | :--- |
| 🟢  | **World Models** (Ha & Schmidhuber, 2018) | VAE + MDN-RNN + Controller | [Read Notes](notes/world-models.md) | [1803.10122](https://arxiv.org/abs/1803.10122) |
| 🟢 | **LeWorldModel** (Maes & LeCun, 2025) | LeWM (JePA) + SIGReg | [Read Notes](notes/lewm.md) | [2501.04690](https://arxiv.org/abs/2501.04690) |
| 🟡 | **V-JEPA** (Meta & LeCun, 2025) | JEPA for Vision Stuff | [Read Notes](notes/vjepa.md) | [2411.05542](https://arxiv.org/abs/2411.05542) |
| ⚪️ | **V-JEPA 2** (Assran et al., 2025) | Self-Supervised Video Models | *Coming Soon* | [2506.09985](https://arxiv.org/abs/2506.09985) |
| ⚪️ | **V-JEPA 2.1** (Mur-Labadia et al., 2025) | Dense Video Representations | *Coming Soon* | [2603.14482](https://arxiv.org/abs/2603.14482) |
| 🟡 | **LeJEPA** (Balestriero & LeCun, 2025) | JEPA + SIGReg | [Read Notes](notes/lejepa.md) | [2501.04690](https://arxiv.org/abs/2501.04690) |
| ⚪️ | **I-JEPA** (Assran et al., 2023) | Vision Transformers, Masking | *Coming Soon* | [2301.08243](https://arxiv.org/abs/2301.08243) |
| ⚪️ | **A-JEPA** (Fei et al., 2025) | Audio Joint-Embedding | *Coming Soon* | [2311.15830](https://arxiv.org/abs/2311.15830) |
| ⚪️ | **Audio-JEPA** (Tuncay et al., 2025) | Audio Representation Learning | *Coming Soon* | [2507.02915](https://arxiv.org/abs/2507.02915) |
| ⚪️ | **M-JEPA** (Teotia et al., 2025) | Audio-Visual Joint-Embedding | *Coming Soon* | [2606.25225](https://arxiv.org/abs/2606.25225) |
| ⚪️ | **Le MuMo JEPA** (Cornelissen et al., 2025) | Multi-Modal Self-Supervised Learning | *Coming Soon* | [2603.24327](https://arxiv.org/abs/2603.24327) |
| ⚪️ | **MC-JEPA** (Bardes et al., 2025) | Motion & Content Features | *Coming Soon* | [2307.12698](https://arxiv.org/abs/2307.12698) |
| ⚪️ | **Echo** (Joint-Embedding for Speech) | Speaker Diarization & Speech Recognition | *Coming Soon* | [2606.01909](https://arxiv.org/abs/2606.01909) |
| ⚪️ | **S-JEPA** (Ioannides et al., 2025) | Speech Representation Learning | *Coming Soon* | [2606.19398](https://arxiv.org/abs/2606.19398) |
| ⚪️ | **ArtNet** (Hu et al., 2025) | Articulatory Predictive Framework | *Coming Soon* | [2606.16595](https://arxiv.org/abs/2606.16595) |
| ⚪️ | **VLA-JEPA** (Sun et al., 2025) | Vision-Language-Action with Latent World Model | *Coming Soon* | [2602.10098](https://arxiv.org/abs/2602.10098) |

*(Status Key: 🟢 Complete | 🟡 In Progress | ⚪️ Planned)*

## 🏗️ Repository Structure

```text
world-model-notes/
├── README.md               # You are here
├── notes/                  # Generated paper summaries and code maps
│   ├── lejepa.md
│   ├── lewm.md
│   ├── vjepa.md
│   ├── world-models.md
│   └── images/             # Diagrams, screenshots, and architecture flows
            
