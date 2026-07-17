# Lets-Do-World-Models 🌍 

The goal of this project is to bridge the gap between heavy theoretical math in research papers and practical engineering implementations in code. It serves as a personal knowledge base and a resource for the broader AI research community.

## 🗂️ Paper Index & Code Maps

Below is the index of papers currently broken down in this repository. Each link leads to a comprehensive markdown file containing an architectural deep-dive, the "useful bits" (hyperparameters, tricks), and a direct mapping of the paper's theory to its GitHub codebase.

| Status | Paper / Architecture | Key Focus | Notes Link |
| :---: | :--- | :--- | :--- |
| 🟢 | **Le World Model** (Maes & LeCun, 2025) | JEPA + SIGReg | [Read Notes](notes/lewm.md) |
| 🟡 | **LeJEPA** (Balestriero & LeCun, 2025) | JEPA + SIGReg | [Read Notes](notes/lejepa.md) |
| 🟡 | **World Models** (Ha & Schmidhuber, 2018) | VAE + MDN-RNN + Controller | [Read Notes](notes/world-models.md) |
| ⚪️ | **I-JEPA** (Assran et al., 2023) | Vision Transformers, Masking | *Coming Soon* |
| ⚪️ | **DreamerV3** (Hafner et al., 2023) | RSSM, Actor-Critic | *Coming Soon* |

*(Status Key: 🟢 Complete | 🟡 In Progress | ⚪️ Planned)*

## 🏗️ Repository Structure

```text
world-model-notes/
├── README.md               # You are here
├── notes/                  # Generated paper summaries and code maps
│   ├── lejepa.md
│   └── world-models.md
└── assets/                 # Diagrams, screenshots, and architecture flows
