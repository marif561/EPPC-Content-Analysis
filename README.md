# EPPC-CARE

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-005BBB?style=flat-square)](https://www.statsmodels.org/)

Analysis code for **EPPC-CARE**: bidirectional content analysis of secure patient–clinical-team messages using the EPPC (Electronic Patient–Provider Communication) codebook. Supports direction-level summaries, BGE-based **content match** for quoted reply pairs, and thread-level models with pseudonymous patient–provider dyads.

This repository contains **scripts only**. It does not include message text, model predictions, summary tables, figures, or study results.

## Repository layout

```
EPPC-Content-Analysis/
├── data/
│   └── README.md
├── scripts/
│   ├── aim2a/
│   │   └── eppc_aim2a_analysis.py
│   ├── content_match/
│   │   └── bge_content_match.py
│   └── threads/
│       ├── parse_threads.py
│       └── adjusted_thread_models.py
├── requirements.txt
└── README.md
```

## Setup

```powershell
git clone https://github.com/marif561/EPPC-Content-Analysis.git
cd EPPC-Content-Analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use a CUDA GPU for `bge_content_match.py` when available.

## Scripts

### Aim 2a direction summaries

```powershell
python scripts/aim2a/eppc_aim2a_analysis.py `
  --messages data/EPPC_Aim2a_Message_Level.csv `
  --annotations data/EPPC_Aim2a_Annotation_Level.csv `
  --out work/results/
```

### Content match (BGE)

```powershell
python scripts/content_match/bge_content_match.py `
  --predictions-jsonl data/eppc_predictions.jsonl `
  --output-csv work/content_match_summary.csv
```

### Thread parsing

```powershell
python scripts/threads/parse_threads.py `
  --thread-export data/thread_export.csv `
  --output-dir work/intermediate/
```

### Adjusted thread models

```powershell
python scripts/threads/adjusted_thread_models.py `
  --threads work/thread_level_dataset.csv
```

## What stays off GitHub

Raw or de-identified message bodies, JSONL predictions, results, figures, linkage keys, and MRNs. See `.gitignore` and `data/README.md`.

## Contributing

Create a branch from `main`, do not commit data or outputs, and open a pull request.

## Citation

Cite the EPPC framework, EPPCMinerBen / EPPC-CARE methods work, and the JAMIA manuscript when published.
