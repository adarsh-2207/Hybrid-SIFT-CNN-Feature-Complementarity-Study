# Hybrid SIFT-CNN Object Recognition
## Experiment Findings Log

---

## Experiment 1 — SIFT Baseline

### Dataset

Caltech-101

- Classes: 101
- Train images: 6,026
- Validation images: 1,256
- Test images: 1,395
- Split: 70/15/15
- Random seed: 42

### SIFT Configuration

- Descriptor: SIFT
- Descriptor dimension: 128
- BoVW vocabulary size: 256
- Vocabulary training descriptors: 100,000
- Classifier: RBF-SVM
- C: 10
- Random seed: 42

### Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 42.52% | 40.86% |
| Macro-F1 | 25.71% | 25.73% |

### Findings

1. SIFT successfully extracted local descriptors from the Caltech-101 images.

2. The SIFT + BoVW + RBF-SVM pipeline achieved 40.86% test accuracy.

3. Performance varied substantially between object classes.

4. Test Macro-F1 (25.73%) was considerably lower than test accuracy (40.86%).

5. Several classes received F1 = 0 because the classifier did not predict them.

6. Some classes performed very well, while others were poorly recognized.

### Research Interpretation

The SIFT baseline demonstrates that handcrafted local features can provide
meaningful object-recognition performance on Caltech-101.

However, the uneven class-level performance indicates limitations in the
SIFT + BoVW representation for representing all object categories equally.

### Baseline Status

COMPLETE / FROZEN

This SIFT configuration will be kept unchanged for the initial CNN and
Hybrid comparisons.

---

## Experiment 2 — CNN Baseline

### Dataset

Caltech-101

- Classes: 101
- Train images: 6,026
- Validation images: 1,256
- Test images: 1,395
- Split: 70/15/15
- Random seed: 42

### CNN Configuration

- Architecture: ResNet-18
- Pretraining: ImageNet
- Input size: 224 × 224
- Batch size: 32
- Epochs: 15
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Data augmentation: Random horizontal flip + random rotation
- Model selection: Best validation Macro-F1

### Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 96.10% | 95.77% |
| Macro-F1 | 93.39% | 94.04% |

### Findings

1. The CNN achieved 95.77% test accuracy and 94.04% test Macro-F1.

2. CNN performance was substantially higher than the SIFT + BoVW + RBF-SVM baseline.

3. CNN performance was relatively consistent across the 101 classes, although some low-sample classes remained more difficult.

4. The strongest difficulties were observed in classes such as crocodile, crocodile_head, water_lilly, platypus, bass, and schooner.

5. The CNN baseline used ImageNet-pretrained ResNet-18 and was fine-tuned on the Caltech-101 training set.

### Comparison with SIFT

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| SIFT + BoVW + RBF-SVM | 40.86% | 25.73% |
| ResNet-18 CNN | 95.77% | 94.04% |

Absolute improvement of CNN over SIFT:

- Accuracy: +54.91 percentage points
- Macro-F1: +68.31 percentage points

### Research Interpretation

On this experimental setup, the learned CNN representation substantially outperformed
the handcrafted SIFT + BoVW representation for Caltech-101 object classification.

However, this result alone does not establish whether SIFT and CNN representations
are complementary. The hybrid experiment is required to determine whether adding
SIFT information provides additional benefit beyond the CNN representation.

### Status

COMPLETE / FROZEN

---

## Experiment 3 — Hybrid SIFT-CNN

### Objective

Determine whether SIFT provides additional discriminative information beyond the CNN representation through feature-level fusion.

### Feature Representations

The experiment used the same fixed train/validation/test split as the SIFT and CNN experiments.

- SIFT representation: 256-D BoVW histogram
- CNN representation: 512-D ResNet-18 penultimate-layer embedding
- Hybrid representation: 768-D concatenation of CNN + SIFT features
- CNN checkpoint: frozen best validation Macro-F1 checkpoint
- SIFT vocabulary: frozen 256-word vocabulary from the SIFT baseline
- Feature standardization: fitted on training data only
- Test set: evaluated only after model selection

### Matched Classifier

To isolate the contribution of SIFT, the same Logistic Regression classifier was applied to:

1. CNN-only 512-D representation
2. CNN + SIFT 768-D representation

This provides a controlled comparison of whether adding SIFT improves the CNN representation.

### Results

| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| CNN matched | 95.78% | 92.87% | 95.48% | 93.67% |
| Hybrid CNN + SIFT | 95.86% | 92.98% | 95.48% | 93.52% |

### Hybrid Improvement

| Metric | Change |
|---|---:|
| Test Accuracy | +0.00 percentage points |
| Test Macro-F1 | -0.15 percentage points |

### Per-Class Analysis

SIFT improved F1 for several classes, including:

- crayfish: +12.0 pp
- wrench: +6.6 pp
- bass: +5.0 pp
- schooner: +4.7 pp
- elephant: +4.0 pp
- crab: +3.9 pp
- menorah: +3.7 pp
- scorpion: +3.2 pp
- ketch: +3.0 pp

However, SIFT also reduced performance for several classes:

- platypus: -10.3 pp
- butterfly: -7.4 pp
- water_lilly: -7.0 pp
- crocodile: -6.9 pp
- windsor_chair: -6.6 pp
- snoopy: -6.6 pp
- flamingo: -4.3 pp
- ibis: -4.3 pp
- chair: -4.3 pp

### Findings

1. SIFT did not provide a net improvement over the matched CNN representation.
2. Test accuracy remained identical at 95.48%.
3. Test Macro-F1 decreased slightly from 93.67% to 93.52%.
4. The class-level analysis indicates that SIFT contains some class-specific information that can improve recognition for certain object categories.
5. However, these improvements were offset by decreases in other classes.
6. Therefore, the tested feature-level SIFT-CNN fusion did not demonstrate overall performance improvement on Caltech-101.

### Research Interpretation

The results do not support the hypothesis that adding the tested SIFT representation to the CNN representation improves overall object-recognition performance on Caltech-101.

However, the per-class results suggest that SIFT may contain localized or class-specific information that overlaps only partially with the CNN representation. In the tested fusion configuration, this information was not sufficient to produce a net improvement.

The result therefore supports a more nuanced conclusion:

> SIFT was substantially weaker than the CNN as a standalone representation, and the tested feature-level fusion did not improve overall performance. Nevertheless, SIFT produced class-specific gains for several categories, indicating partial but insufficient complementarity.

### Status

COMPLETE / FROZEN

SIFT Hyperparameter Tuning — FINAL / FROZEN

A bounded validation-based search was performed over:
C = {1, 10, 100}
gamma = {scale, 0.001, 0.003, 0.01}

Selection criterion: Validation Macro-F1.

Best configuration:
C = 10
gamma = scale

Validation:
Accuracy = 42.83%
Macro-F1 = 25.20%

Final test:
Accuracy = 40.36%
Macro-F1 = 25.58%

The selected configuration is identical to the original SIFT baseline configuration.
Therefore, the original SIFT configuration is retained as the final frozen SIFT model.

No further SIFT hyperparameter tuning will be performed in the current study.
