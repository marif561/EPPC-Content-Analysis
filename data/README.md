# Data (not in git)

Secure patient–provider messages and model prediction files are **not** stored in this repository. Place approved files on your machine only, under `data/` or another local path, and pass paths to the scripts on the command line.

Typical inputs (names may vary locally):

| Kind | Used by |
|------|---------|
| Message-level and annotation-level EPPC tables | `scripts/aim2a/eppc_aim2a_analysis.py` |
| EPPC-CARE predictions JSONL (one message per line) | `scripts/content_match/bge_content_match.py` |
| Portal thread export CSV | `scripts/threads/parse_threads.py` |
| Pseudonymous thread-level summary CSV (built locally) | `scripts/threads/adjusted_thread_models.py` |

Intermediate pickles with message bodies should go under `work/` (git-ignored).
