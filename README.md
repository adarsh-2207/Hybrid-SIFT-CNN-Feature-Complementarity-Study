# Hybrid SIFT-CNN Object Recognition

### Investigating the Complementarity of Local SIFT Features and Deep CNN Representations for Image Classification

## Overview

Hybrid SIFT-CNN Object Recognition is a computer vision research project that investigates whether handcrafted local features extracted using the **Scale-Invariant Feature Transform (SIFT)** can complement deep visual representations learned by **Convolutional Neural Networks (CNNs)** for multi-class image classification.

The project combines SIFT-based Bag of Visual Words (BoVW) representations with CNN feature embeddings and evaluates different feature-fusion and classification strategies on the **Caltech-101 and Caltech-256 datasets**.

Rather than assuming that combining different feature representations improves performance, this project experimentally examines whether local keypoint-based information provides additional discriminative value beyond CNN features.

> **Principal finding:** Under the tested SIFT-BoVW representations, classifiers, and fusion strategies, consistent improvements over CNN-only baselines were not observed. This highlights an important research question in hybrid computer vision: combining different feature representations does not necessarily produce complementary predictive information.

---

## Research Objectives

The primary objective is to investigate whether handcrafted local features extracted using SIFT provide complementary information to deep CNN features for object recognition.

The project focuses on the following configurations:

| Configuration | Representation | Objective |
|---|---|---|
| CNN-only | Deep CNN embeddings | Establish a deep-learning baseline |
| SIFT-only | SIFT with BoVW | Evaluate handcrafted local features independently |
| Hybrid | CNN and SIFT features | Investigate the effectiveness of feature fusion |

The central hypothesis is that SIFT captures local keypoint structures that may complement the higher-level visual representations learned by CNNs.

The experiments evaluate this hypothesis rather than assuming it to be true.

---

## System Architecture

```text
                  Input Images
                       |
              Dataset Preparation
                       |
              Train / Val / Test
                       |
          +------------+------------+
          |                         |
          v                         v
     SIFT Pipeline              CNN Pipeline
          |                         |
     Keypoint Detection       Deep Feature Extraction
          |                         |
     Descriptor Extraction    CNN Embeddings
          |                         |
     Visual Vocabulary              |
          |                         |
     BoVW Histograms                |
          |                         |
          +------------+------------+
                       |
                 Feature Fusion
                       |
              +--------+--------+
              |                 |
         Feature-Level      Decision-Level
            Fusion              Fusion
              |                 |
              +--------+--------+
                       |
                 Classification
                       |
                 Model Evaluation
                       |
          Accuracy / Macro-F1 / Weighted-F1
                       |
              Statistical Analysis
                       |
               Final Comparison
```

---

## Methodology

### 1. Dataset Preparation

The project uses two benchmark image-classification datasets.

| Dataset | Categories | Images |
|---|---:|---:|
| Caltech-101 | 101 | 8,677 |
| Caltech-256 | 256 | 30,607 |

For Caltech-256, the clutter category is excluded, resulting in 29,780 images across 256 object categories.

The primary Caltech-256 experimental split contains:

| Split | Images |
|---|---:|
| Training | 20,841 |
| Validation | 4,467 |
| Testing | 4,472 |
| **Total** | **29,780** |

A separate earlier MLP experiment uses a different split. Its results should not be directly compared with experiments using the primary split.

### 2. SIFT Feature Extraction

SIFT is used to detect distinctive local keypoints and extract descriptors around them.

The extracted descriptors represent local image structures and are used to construct a fixed-dimensional visual representation.

### 3. Bag of Visual Words

Since images contain varying numbers of SIFT descriptors, a visual vocabulary is constructed using clustering.

The pipeline consists of:

- Extracting SIFT descriptors from training images.
- Sampling descriptors to control memory consumption.
- Learning a visual vocabulary using MiniBatch K-Means.
- Assigning descriptors to visual words.
- Constructing normalized BoVW histograms.

The vocabulary is learned using training data only to avoid information leakage.

### 4. CNN Feature Extraction

CNNs provide deep visual representations that capture increasingly complex image patterns.

The project includes experiments involving:

- ResNet-18
- MobileNetV3-Small

Depending on the experiment, CNN representations are obtained through pretrained feature extractors or fine-tuned classification models.

### 5. Hybrid Feature Fusion

The project investigates different approaches to combining CNN and SIFT representations.

**Feature-Level Fusion**

CNN and SIFT feature vectors are combined into a joint representation before classification.

**Weighted Feature Fusion**

The relative contribution of CNN and SIFT representations is varied to investigate whether different feature weights improve classification.

**Late Fusion**

Separate CNN and SIFT classifiers produce predictions that are combined at the decision level.

**MLP-Based Fusion**

Multilayer Perceptrons are used to investigate nonlinear classification and fusion of the feature representations.

**PCA-Based Fusion**

Dimensionality reduction is investigated to evaluate whether a lower-dimensional SIFT representation can improve hybrid classification.

**Temperature-Scaled Late Fusion**

Probability calibration is explored before combining classifier outputs.

### 6. Evaluation and Statistical Analysis

The project evaluates models using:

- Classification accuracy
- Macro-F1 score
- Weighted-F1 score
- Per-class precision and recall
- Confusion matrices
- Class-wise error analysis
- Prediction disagreement analysis
- McNemar's statistical test
- Paired bootstrap confidence intervals

The validation set is used for model and fusion selection, while the test set is used for final evaluation.

---

## Experimental Results

### Caltech-101

Matched frozen ResNet-18 feature experiment using Logistic Regression (C=1):

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| CNN-only | 95.48% | 93.67% |
| CNN + SIFT | 95.48% | 93.52% |

The hybrid representation did not improve test accuracy and produced a slightly lower Macro-F1 score.

Additional experiments involving weighted feature fusion and late fusion selected zero contribution from SIFT.

### Caltech-256

Matched frozen MobileNetV3-Small feature experiment using Logistic Regression (C=1):

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| CNN-only | 77.08% | 75.15% |
| SIFT-only | 13.75% | 10.79% |
| CNN + SIFT | 77.01% | 74.98% |

The hybrid representation produced a small decrease relative to the CNN-only baseline.

### SIFT + BoVW + RBF-SVM Baseline

| Dataset | Test Accuracy | Macro-F1 |
|---|---:|---:|
| Caltech-101 | 40.86% | 25.73% |
| Caltech-256 | 19.77% | 14.57% |

These results demonstrate the performance of the standalone SIFT-BoVW pipeline under the tested configurations.

**Note:** Results from different CNN architectures, feature extractors, classifiers, and dataset splits represent separate experiments and should not be interpreted as direct head-to-head comparisons.

---

## Further Analysis

Beyond aggregate accuracy, the project investigates whether SIFT helps on individual images or object categories.

The analysis includes:

- CNN and SIFT prediction disagreement.
- CNN-correct/SIFT-wrong and CNN-wrong/SIFT-correct cases.
- Class-wise performance improvements and declines.
- Fusion error correction and error introduction.
- Statistical significance of paired prediction differences.
- Effects of feature weighting and dimensionality reduction.

The overall findings suggest that SIFT's standalone predictive performance is substantially lower than that of the tested CNN representations. Under the investigated configurations, its contribution was insufficient to produce reliable improvements through fusion.

This does not establish that SIFT is universally ineffective. Different local descriptors, encodings, architectures, and fusion mechanisms may produce different outcomes.

---

## Technology Stack

**Programming**
- Python
- Jupyter Notebook

**Computer Vision**
- OpenCV
- SIFT
- Bag of Visual Words

**Deep Learning**
- PyTorch
- Torchvision
- ResNet-18
- MobileNetV3-Small

**Machine Learning**
- Scikit-learn
- Logistic Regression
- Support Vector Machines
- MiniBatch K-Means
- Multilayer Perceptrons
- Principal Component Analysis

**Data Analysis**
- NumPy
- Pandas
- Matplotlib
- SciPy

---

## Repository Structure

```text
Hybrid-SIFT-CNN-Object-Recognition/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── cnn/
│   │   ├── cnn_baseline.py
│   │   ├── extract_cnn_features.py
│   │   ├── extract_cnn_features_caltech256.py
│   │   ├── train_mobilenet.py
│   │   └── ...
│   │
│   ├── sift/
│   │   ├── sift_bovw_svm.py
│   │   ├── sift_bovw_svm_caltech256.py
│   │   ├── extract_bovw_features.py
│   │   ├── sift_svm_tuning.py
│   │   └── ...
│   │
│   ├── data/
│   │   └── prepare_caltech256.py
│   │
│   ├── hybrid_train.py
│   ├── late_fusion_experiment.py
│   ├── late_fusion_mlp_caltech256.py
│   ├── pca_hybrid_fusion_caltech256.py
│   ├── train_mlp_caltech256.py
│   ├── weighted_mlp_caltech256.py
│   └── ...
│
├── results/
│   ├── cnn/
│   ├── sift/
│   ├── hybrid/
│   ├── final/
│   ├── late_fusion_caltech101/
│   └── project_findings.md
│
├── results_caltech256/
│   ├── classwise_analysis/
│   ├── hybrid_classifier/
│   ├── hybrid_features/
│   ├── mlp_updated/
│   ├── late_fusion/
│   ├── late_fusion_mlp/
│   ├── weighted_fusion/
│   ├── weighted_mlp/
│   ├── pca_hybrid_fusion/
│   └── statistical_evaluation/
│
├── Hybrid_SIFT_CNN_Full_Project_Report_Revised.docx
├── SIFT_CNN_AI_Agent_Full_Context.pdf
├── SIFT M CNN.pdf
└── Untitled0 (1).ipynb
```

Dataset directories, model checkpoints, generated feature arrays, and large intermediate artifacts are not included in the Git repository.

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/adarsh-2207/Hybrid-SIFT-CNN-Object-Recognition.git

cd Hybrid-SIFT-CNN-Object-Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn opencv-contrib-python torch torchvision jupyter
```

OpenCV's contrib package provides the SIFT functionality used by the project.

### 4. Download and Prepare the Datasets

Download Caltech-101 and/or Caltech-256 and prepare the directory structure expected by the relevant scripts.

Expected paths include:

```text
project-root/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data_caltech256/
    ├── train/
    ├── val/
    └── test/
```

The datasets are not included in this repository.

### 5. Run Experiments

For the Caltech-256 SIFT-BoVW-SVM baseline:

```bash
python src/sift/sift_bovw_svm_caltech256.py
```

For the Caltech-101 SIFT-BoVW-SVM baseline:

```bash
python src/sift/sift_bovw_svm.py
```

Additional experiment scripts are available under `src/`.

Some scripts depend on feature files or intermediate results generated by earlier pipeline stages. Follow the relevant experiment's data preparation and feature extraction steps before running it.

---

## Limitations

The current research implementation has several limitations:

- SIFT-BoVW performance is considerably lower than the tested CNN baselines.
- The investigated fusion strategies did not establish reliable complementary gains.
- Results depend on the chosen CNN architecture, SIFT vocabulary, classifier, and fusion method.
- Some experiments use different dataset splits and are not directly comparable.
- Feature extraction and model training can require considerable memory and computation.
- The current experiments do not establish that all possible SIFT-CNN combinations are ineffective.

These limitations provide opportunities for further investigation.

---

## Future Work

Potential research directions include:

- Investigating alternative local feature descriptors.
- Exploring advanced visual encoding techniques such as VLAD and Fisher Vectors.
- Testing attention-based feature fusion.
- Exploring spatially aware feature representations.
- Investigating feature selection and dimensionality reduction.
- Evaluating stronger CNN and Vision Transformer architectures.
- Testing additional datasets and more diverse object-recognition benchmarks.
- Improving computational efficiency and reproducibility.
- Investigating the relationship between feature complementarity and class-specific recognition performance.

These are potential research directions and are not claims about completed experiments.

---

## Conclusion

The objective of this project is not simply to combine traditional computer vision with deep learning, but to experimentally investigate whether their representations provide complementary information.

By comparing standalone and hybrid approaches, this work explores the strengths and limitations of handcrafted and learned visual representations in multi-class object recognition.

The experimental findings demonstrate that feature fusion does not inherently guarantee improved classification performance and that its effectiveness must be established through controlled evaluation.

---

## Author

**Adarsh Bhaskar**  
B.Tech — Computer Science & Engineering (IoT and Intelligent Systems) 

Manipal University Jaipur

Research interests:
- Computer Vision
- Machine Learning
- Deep Learning
- Feature Engineering
- Hybrid AI Systems

---

## License

This project is released under the license included in the repository. See [LICENSE](LICENSE) for details.
