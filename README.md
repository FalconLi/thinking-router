# Thinking Router

**Learning When to Reason: Lightweight Routing for a Small Language Model**

## Setup

Use Linux x86-64 and one NVIDIA GPU with BF16 support.
The reference GPU is an RTX 5090 (32 GB); the NVIDIA driver must support CUDA 12.8.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed;
it supplies Python 3.12 when creating the environment below.

```bash
git clone https://github.com/FalconLi/thinking-router.git
cd thinking-router
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements-gpu.lock
uv pip install --no-deps --no-build-isolation -e .

# Optional: display GPU usage and running processes to help choose a GPU.
nvidia-smi

# Select physical GPU 0; this program will refer to it as cuda:0.
export CUDA_VISIBLE_DEVICES=0
# Optional: limit OpenMP CPU workers to four during numerical computations.
export OMP_NUM_THREADS=4
# Optional: limit OpenBLAS matrix operations to one CPU thread to avoid excess CPU load.
export OPENBLAS_NUM_THREADS=1

# Unpack and verify the included GSM8K questions and fixed splits into data/.
python -m routing_project.prepare_data

# Download the model and tokenizer to this machine, normally ~/.cache/huggingface/hub/.
# The revision fixes the exact Qwen3-1.7B version used by the experiments.
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
# Optional after download: load local files without contacting Hugging Face.
export HF_HUB_OFFLINE=1
```

`datasets/gsm8k.tar.gz` includes 2,000 training, 500 validation, and 1,319 test
questions from `openai/gsm8k`, plus the fixed training subsets and 100-question
timing subset.

## Reproduce E1–E4

Run in a persistent terminal with the GPU reserved for the run:

```bash
python -u scripts/run_experiments.py --session session-1
```

The command generates Qwen responses and prompt features, trains all 32 classifiers,
selects policies on validation data, freezes them before using test answers, and runs:

| Experiment | Measurement |
| --- | --- |
| E1 | Accuracy and output tokens with thinking always on or always off |
| E2 | Hidden-state, TF-IDF, question-length, and random routing comparisons |
| E3 | Classifier performance with 200, 500, 1,000, and 2,000 training questions |
| E4 | Live latency of always-thinking, never-thinking, hidden-state, and TF-IDF policies |

To run the two stages separately:

```bash
python -u scripts/run_experiments.py --experiments e1-e3
python -u scripts/run_experiments.py --experiments e4 --session session-1
```

Rerun the same command after an interruption to reuse completed responses and
features. Use a new `--session` name if E4 timing conditions change. E4 monitors
the GPU and stops if another CUDA process appears. Allow several hours for a full
run.

`configs/experiment.yaml` specifies model/dataset revisions, seeds, sampling,
training sizes, and evaluation settings. Generated files are grouped by run ID:

- `data/`: unpacked questions/splits, generated responses, and hidden-state arrays.
- `artifacts/`: trained classifiers and validation-selected policies.
- `results/<run_id>/`: CSV tables (`tables/`, `curves/`), PDF/PNG figures (`figures/`),
  and per-question E4 timings (`raw/`).

These generated directories are ignored by Git. Rebuild tables and figures from
an existing run with `python -m routing_project.evaluate` and
`python -m routing_project.make_figures`.
