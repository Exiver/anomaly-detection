# Model card: PatchCore anomaly detector

## Intended use
Unsupervised visual inspection research and demonstration for the MVTec AD categories included in a release. The model learns a reference distribution from normal training images and returns an image-level anomaly score, a threshold-based verdict, and a pixel anomaly map. The v2 service release contains bottle, carpet, and hazelnut models.

## Out of scope
This benchmark model is not validated for production quality control, safety decisions, other datasets, or product categories absent from the release. Do not use its verdict as the sole basis for rejecting products. Validate on representative images from the target cameras, environment, and product variants before any operational use.

## Data and license
The models were trained and calibrated with the MVTec AD dataset. MVTec AD is licensed separately under CC BY-NC-SA 4.0, and its official terms prohibit commercial use. The project source code is MIT-licensed; that code license does not grant rights to MVTec data or artifacts trained from it. Treat these model artifacts as non-commercial unless MVTec confirms otherwise. See the [MVTec AD dataset terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

## Configuration and thresholds
<!-- BEGIN AUTO -->
- **Release:** v2 (built 2026-11-15T08:57:48.980013+00:00)
- **Preprocessing:** resize to 256x256, center crop 224, ImageNet normalization
- **Embedding:** WideResNet50-2 layer2 + layer3, 1024-dimensional patch embeddings, coreset ratio 0.1
- **Map smoothing:** Gaussian blur, sigma 4.0
- **Threshold rule:** max, calibrated on a 10% holdout of normal training images
- **Seed:** 42

| category | threshold | holdout images | bank size | test image AUROC | test pixel AUROC | recall @ threshold | FPR @ threshold |
|---|---|---|---|---|---|---|---|
| bottle | 1.312 | 20 | 14817 | 1.000 | 0.986 | 1.000 | 0.050 |
| carpet | 1.201 | 28 | 19756 | 0.984 | 0.990 | 0.989 | 0.429 |
| hazelnut | 2.031 | 39 | 27596 | 1.000 | 0.986 | 0.957 | 0.000 |

_Test metrics are informational; the thresholds were selected without test labels._
<!-- END AUTO -->

## Known failure modes
The full benchmark evaluation shows weaker defect separation for pill cracks, grid glue, and capsule scratches. Those categories are not included in the current v2 serving release. Within v2, the saved test evaluation shows a substantially higher false-positive rate for carpet than for bottle or hazelnut at the listed thresholds. These measurements come from the checked-in evaluation artifacts and need to be rerun before being treated as results for the current source: the artifacts do not record the preprocessing revision, and earlier notes describe an evaluation transform without ImageNet normalization.

## Monitoring
The service exposes health/readiness endpoints and Prometheus metrics. Monitor request volume, latency, error rates, anomaly-score distributions, and drift signals by category. Drift signals use unlabeled traffic and a small reference set: they can indicate a distribution shift but cannot measure live accuracy. Periodically review predictions against labeled, representative samples and recalibrate thresholds when the operating data changes.

## Limitations
MVTec AD is a small, controlled benchmark and does not represent production variation in cameras, lighting, motion, or product revisions. Each release category has its own model; categories outside the release are unsupported. The threshold table is based on a 10% holdout of normal training images, but only three categories are present in v2, and the observed carpet false-positive rate illustrates that a holdout threshold does not guarantee a desired operational error rate. Evaluation metrics are based on saved CSVs whose preprocessing provenance is incomplete; rerun training and evaluation with the current pipeline before making performance claims. Results and speed measurements do not establish production reliability or transfer to other hardware.
