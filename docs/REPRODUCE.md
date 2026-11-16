# Reproducing the results

The generated tables are in [`reports/RESULTS.md`](../reports/RESULTS.md). The environment used to generate the environment report is recorded in [`reports/environment.txt`](../reports/environment.txt); the package lock snapshot is [`requirements-lock.txt`](../requirements-lock.txt). The project seed is recorded in the release manifest.

**Metric provenance caveat:** the saved Phase 4 notes describe an evaluation transform without ImageNet normalization, while the current `MVTecDataset` uses the normalized transform in `src/anomaly/data/transforms.py`. The CSVs do not record their preprocessing revision. `RESULTS.md` faithfully formats the saved CSVs, but rerun training and evaluation below before treating those values as results of the current source.

## Environment and data

Use Python 3.11 and install the CPU or CUDA build of PyTorch appropriate for the machine before installing the project extras:

```powershell
conda create -n ai-engineering python=3.11
conda activate ai-engineering
python -m pip install --upgrade pip
# Install the matching PyTorch and torchvision build from https://pytorch.org/get-started/locally/
python -m pip install -e ".[train,service,dev]"
```

Download MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) into `data/mvtec_ad/`, preserving its category, `train`, `test`, and `ground_truth` directory layout. The dataset page states that the data is under CC BY-NC-SA 4.0 and prohibits commercial use. The dataset is not included here.

## Training and evaluation

Run from the repository root after installing dependencies and placing the data:

```powershell
python scripts/run_ae.py
python scripts/run_patchcore.py
python scripts/evaluate_model.py --method ae
python scripts/evaluate_model.py --method patchcore
```

These commands train/evaluate the full category set. `run_ae.py` accepts `--skip-train` when checkpoints already exist; `run_patchcore.py` accepts `--skip-fit` when memory banks already exist. The evaluator saves category metrics, per-defect AUROCs, and error galleries in `reports/`.

## Calibration and performance

```powershell
python scripts/percision_experiment.py --category bottle
python scripts/calibrate_threshold.py --categories bottle carpet
```

The first command writes the available bottle threshold calibration. The second evaluates the precision variants on bottle and carpet and writes `reports/precision_experiment.csv`. The script names reflect the current repository; `percision_experiment.py` is the threshold-calibration script.

## Framework comparison

The in-house prediction export is:

```powershell
python scripts/export_predictions.py --method patchcore
```

Run the anomalib exporter in an environment with anomalib installed, then compare the aligned predictions:

```powershell
python scripts/anomaly_export.py --data-root data/mvtec_ad --run-name anomalib_patchcore
python scripts/compare_frameworks.py --a patchcore --b anomalib_patchcore
```

The repository has two framework-comparison CSVs, including a `k1` run. The exact CLI parameters for that second anomalib run are not recorded in the report artifacts, so it must be rerun and its parameters recorded before claiming an exact reproduction.

## Release and service

The current v2 release manifest covers bottle, carpet, and hazelnut:

```powershell
python scripts/build_release.py --version v2 --categories bottle carpet hazelnut
uvicorn anomaly.service.main:app --host 0.0.0.0 --port 8000
```

The service demo is at `/demo`; the API schema is at `/docs`. A local API key can be set as `AD_API_KEY` in an ignored `.env` file. Never commit real keys.

## Fault injection

`scripts/load_test.py` accepts `--category`, `--image-dir`, `--requests`, and `--concurrency`. Use `--brightness 0.5` for the dark-image input fault and `--only-defects` for an anomaly-rate shift. The checked-in reports do not preserve the exact request counts, concurrency values, and command lines from the previous fault runs; rerun them and save the exact commands and resulting Prometheus screenshots before treating those runs as reproducible evidence.

## Generate documentation

```powershell
python scripts/env_report.py
python scripts/make_docs.py --version v2
python -m pip freeze > requirements-lock.txt
```

`make_docs.py` writes `reports/RESULTS.md`, refreshes the generated results table in the README, and updates only the generated configuration block in `MODEL_CARD.md`. It labels the confidence intervals according to the evaluator's current `alpha=0.005` setting (99.5% intervals); they must not be described as 95% intervals.

## Tests and lint

```powershell
python -m pytest -p no:cacheprovider -q
ruff check .
```

The cache-provider option avoids writing pytest's cache in environments where that directory is not writable.
