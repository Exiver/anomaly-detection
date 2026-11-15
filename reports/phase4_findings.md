# Phase 4 findings

## 1. Setup and evidence sources

The evaluation uses all 15 MVTec AD categories. The repository's dataset summary counts 3,629 defect-free training images and 1,725 test images (467 good and 1,258 anomalous); each test image has a pixel mask for defects. Per-category test sizes range from 42 to 167 images. Images are converted to RGB, resized to 256×256, center-cropped to 224×224, and converted to `[0, 1]` tensors. Masks use nearest-neighbor resizing and the same crop. The pipeline applies no image normalization or augmentation.

Both methods use seed 42, 224×224 model inputs, and image-level scores equal to the maximum pixel anomaly score. **Autoencoder (AE):** one model per category, trained on normal images with an MSE reconstruction loss; config defaults are latent channels 64, batch size 16, up to 100 epochs, Adam at 0.001, 10% validation split, and patience 10. **PatchCore:** one memory bank per category, using a frozen ImageNet-pretrained WideResNet-50-2, layer 2/3 patch features, embedding dimension 1,024, coreset ratio 0.1, nearest-neighbor chunk 2,048, and Gaussian blur sigma 4. These are the code defaults in `src/anomaly/config.py`; the report directory does not record a run-specific config snapshot.

`evaluation_ae.csv` and `evaluation_patchcore.csv` contain image AUROC, pixel AUROC, PRO, and test counts; `per_defect_*.csv` contains image AUROC for each defect type versus good images. The AUROC 95% confidence intervals below were recalculated from the saved models' test image scores using 1,000 seeded bootstrap resamples (`seed=42`, `alpha=0.05`). The existing CSV `ci_low`/`ci_high` columns were generated with `alpha=0.005`, which yields 99.5% rather than 95% intervals, so those values are not used below. Macro means are unweighted arithmetic means over the 15 category scores; the CI cells in the mean row are intentionally blank because a mean of category endpoints is not a pooled confidence interval.

Threshold results come from `threshold_calibration_bottle.csv` and `scripts/calibrate_threshold.py`. The precision study comes from `precision_experiment.csv` and `scripts/percision_experiment.py`; that experiment covers only bottle and carpet. The comparison CSV is `method_comparison.csv`.

## 2. Results by category

Image AUROC is shown with its recalculated 95% bootstrap CI. Pixel AUROC and PRO are point estimates. Means are macro averages; higher is better for all three metrics.

| Category | AE image AUROC (95% CI) | AE pixel AUROC | AE PRO | PatchCore image AUROC (95% CI) | PatchCore pixel AUROC | PatchCore PRO |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 0.9135 (0.8507–0.9683) | 0.7454 | 0.5495 | 1.0000 (1.0000–1.0000) | 0.9797 | 0.9298 |
| cable | 0.5352 (0.4440–0.6246) | 0.6485 | 0.4048 | 0.9912 (0.9795–0.9980) | 0.9840 | 0.9371 |
| capsule | 0.6725 (0.5638–0.7740) | 0.9224 | 0.7286 | 0.9625 (0.9060–0.9944) | 0.9845 | 0.9035 |
| carpet | 0.6730 (0.5603–0.7794) | 0.7501 | 0.5218 | 0.9679 (0.9380–0.9909) | 0.9869 | 0.9160 |
| grid | 0.9774 (0.9440–0.9984) | 0.9342 | 0.8866 | 0.9799 (0.9378–1.0000) | 0.9846 | 0.9224 |
| hazelnut | 0.9718 (0.9423–0.9915) | 0.9778 | 0.8535 | 0.9996 (0.9979–1.0000) | 0.9856 | 0.8647 |
| leather | 0.7687 (0.6843–0.8387) | 0.9654 | 0.8976 | 0.9969 (0.9903–1.0000) | 0.9922 | 0.9581 |
| metal_nut | 0.5024 (0.3637–0.6220) | 0.6489 | 0.3572 | 0.9980 (0.9924–1.0000) | 0.9835 | 0.9159 |
| pill | 0.6669 (0.5676–0.7681) | 0.9019 | 0.8346 | 0.9318 (0.8859–0.9701) | 0.9574 | 0.9198 |
| screw | 0.6969 (0.5942–0.7940) | 0.9556 | 0.8271 | 0.9693 (0.9227–0.9986) | 0.9942 | 0.9679 |
| tile | 0.7237 (0.6360–0.8107) | 0.5647 | 0.4591 | 0.9913 (0.9766–1.0000) | 0.9372 | 0.7725 |
| toothbrush | 0.9750 (0.9045–1.0000) | 0.9275 | 0.7990 | 1.0000 (1.0000–1.0000) | 0.9856 | 0.8780 |
| transistor | 0.5975 (0.4708–0.7201) | 0.7277 | 0.5307 | 0.9979 (0.9919–1.0000) | 0.9513 | 0.9127 |
| wood | 0.9439 (0.8897–0.9848) | 0.7859 | 0.6666 | 0.9904 (0.9709–1.0000) | 0.9426 | 0.8599 |
| zipper | 0.8713 (0.7924–0.9388) | 0.8286 | 0.5591 | 0.9921 (0.9814–0.9994) | 0.9868 | 0.9470 |
| **Macro mean** | **0.7660 (—)** | **0.8190** | **0.6584** | **0.9846 (—)** | **0.9757** | **0.9070** |

## 3. Where PatchCore wins and where the evidence overlaps

PatchCore has a higher image AUROC point estimate in all 15 categories. Using the requested rule of not claiming a clear difference when the category-level 95% intervals overlap, the intervals do not overlap for 12 categories: bottle, cable, capsule, carpet, hazelnut, leather, metal nut, pill, screw, tile, transistor, and zipper. They overlap for grid, toothbrush, and wood, so these results do not establish a clear image-AUROC separation there under this interval check. This is a descriptive overlap check, not a paired statistical test.

The macro means favor PatchCore by 0.2186 image AUROC, 0.1568 pixel AUROC, and 0.2486 PRO. The largest macro gap is region-level PRO, consistent with PatchCore's stronger localization on most categories. Tile is a notable exception in localization: PatchCore's PRO is 0.7725, much lower than its other categories, though still above AE's 0.4591.

## 4. Failure modes and visual evidence

The error galleries select the lowest-scored anomalous images (misses) and highest-scored good images (false alarms). Defect-level AUROC compares one defect type against good images; it is not a defect-only classification accuracy.

1. **AE misses cable defects, especially missing components.** Cable image AUROC is 0.5352 (95% CI 0.4440–0.6246). `missing_cable` scores 0.3376, `missing_wire` 0.3655, and `poke_insulation` 0.3914 in the per-defect file. The miss row in [the AE cable error gallery](errors_ae_cable.png) includes missing-cable examples whose image scores are near zero; its good-image row also shows high-scoring normal cable assemblies.
2. **AE struggles with metal-nut surface changes.** Metal-nut image AUROC is 0.5024 (95% CI 0.3637–0.6220); `scratch` is 0.3300 and `color` is 0.4008. [The AE metal-nut gallery](errors_ae_metal_nut.png) shows missed scratches and color defects, with localization responses concentrated on small surface/edge areas.
3. **Tile textures remain a localization weakness, especially for PatchCore.** PatchCore tile image AUROC is 0.9913 but pixel AUROC is 0.9372 and PRO is 0.7725, the lowest PatchCore PRO. `gray_stroke` is the weakest PatchCore tile defect type at 0.9583 image AUROC. [The PatchCore tile gallery](errors_patchcore_tile.png) shows missed gray-stroke/rough cases and diffuse maps across the repeated texture. AE is weaker still on tile localization (pixel AUROC 0.5647, PRO 0.4591); its `gray_stroke` defect AUROC is 0.3788 in [the AE tile gallery](errors_ae_tile.png).
4. **Normal object details can attract anomaly heat.** The high-scoring good examples in [the AE cable gallery](errors_ae_cable.png) show bright responses around cable terminals. This is qualitative evidence of a potential false-alarm pattern; the gallery does not define a deployment threshold, so it should not be interpreted as a false-positive rate.

**Gallery rendering caveat:** the dataset loader returns `[0, 1]` image tensors without ImageNet normalization, while `viz.denormalize()` applies an ImageNet inverse-normalization. Thus gallery RGB colors are shifted. The sample ranks, masks, and anomaly maps remain from the evaluation, but use the galleries cautiously for fine-grained visual or color judgments; fix the visualization before publishing them as polished figures.

## 5. Operating threshold: bottle only

The available calibration CSV covers **bottle only**. Both normal-only calibration rules produced the same reported test result:

| Rule | Threshold | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Maximum normal holdout score | 1.0765 | 0.9844 | 1.0000 | 0.9921 | 0.0500 |
| Mean + 3 standard deviations | 1.1320 | 0.9844 | 1.0000 | 0.9921 | 0.0500 |
| Oracle best-F1 threshold (uses test labels) | 1.2342 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

On this bottle test split, the calibrated rules retain full recall and make one false positive among 20 good images (5% FPR). The test-label oracle removes that one false positive, a 0.0079 F1 improvement, but is not deployable because it uses the answers it is meant to predict. This single-category result does not show that the same threshold behavior will transfer to the other 14 categories; each category needs calibration and validation on its own representative normal data.

## 6. Precision, latency, and memory study

The precision experiment covers bottle and carpet only, on an NVIDIA GeForce RTX 4050 Laptop GPU. Values below are means over those two categories; image AUROC, pixel AUROC, and PRO are unchanged to the reported precision across variants.

| Variant | Mean image AUROC | Mean pixel AUROC | Mean PRO | Mean latency (ms/image) | Mean peak GPU allocation (MB) | Mean stored bank (MB) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.9840 | 0.9833 | 0.9229 | 11.7 | 514.0 | 78.5 |
| FP16 backbone | 0.9838 | 0.9833 | 0.9229 | 9.9 | 495.0 | 78.5 |
| FP16 backbone + FP16 bank | 0.9838 | 0.9833 | 0.9229 | 9.8 | 534.5 | 39.3 |

On these two categories, mixed-precision inference was about 15–16% faster than FP32, with no meaningful change in the rounded quality metrics. Storing the bank in FP16 halves its stored size (67.1→33.6 MB for bottle; 89.9→45.0 MB for carpet). That storage reduction does **not** imply lower peak runtime allocation: the FP16-bank variant averaged 534.5 MB peak versus 514.0 MB for FP32, and carpet rose from 541 to 586 MB. Peak allocation includes runtime/model/workspace memory, whereas `bank_mb` measures only the memory-bank tensor. Latency and peak-memory results are hardware- and workload-specific, based on a warm-up and five timed repetitions over a sample batch; they are not general hardware guarantees.

## 7. Limitations and interpretation

- MVTec AD is one benchmark with 15 object/texture categories, standardized framing, and clean capture conditions; results do not establish robustness to production lighting changes, camera changes, motion, or domain shift. The official dataset page describes the 15-category benchmark and pixel annotations: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
- Test sets are small by category (42–167 images here), and some defect types contain few examples. The bootstrap intervals reflect image-level sampling uncertainty on these test sets, not uncertainty across factories, cameras, or production lines.
- The macro means weight every category equally. They are not image-weighted or pooled pixel-level metrics.
- Threshold calibration is reported only for bottle. Its oracle row is an optimistic, test-label-dependent reference, not a valid deployment threshold.
- The precision study tests two categories and one GPU; its latency and memory results should not be generalized beyond this setup.
- The evaluation code uses an ImageNet-pretrained PatchCore backbone, but the dataset transform currently returns `[0, 1]` pixels without ImageNet mean/std normalization. Re-run with a preprocessing choice matched to the pretrained backbone before treating these as final model-selection numbers.
- MVTec AD is distributed under CC BY-NC-SA 4.0; the official page prohibits commercial use under its terms. Check the dataset terms before redistributing data or using it in a commercial context.
