# Bone Age Prediction from Hand Radiographs


---


## Project Overview

Bone age assessment is a crucial clinical task used in pediatrics to evaluate skeletal maturity, growth disorders, and endocrine abnormalities. Traditionally performed manually by radiologists, this process is time-consuming and subjective.

This project aims to **automate bone age prediction from hand X-ray images** using machine learning and deep learning techniques.  
We address the problem in two complementary ways:

- **Regression**: Predict the exact bone age (continuous value)
- **Classification**: Predict discrete developmental stages

Both **classical ML** and **deep learning** approaches are explored, compared, and analyzed with explainability techniques.

---


## Dataset

- **Dataset**: RSNA Bone Age Dataset
- **Input**: Hand X-ray images
- **Target (Regression)**: Bone age in months / years
- **Target (Classification)**: Bone age category (5 classes)
- **Metadata**: Biological sex (Male / Female)

Images are named as `<id>.png` and mapped to metadata via CSV files.

---

## Data Preprocessing

- Image resizing to fixed resolution
- ROI enhancement using contrast stretching
- Normalization using ImageNet statistics
- Train / Validation / Test split (70 / 15 / 15)
- Sex encoded and used as an auxiliary model input

---

## Approaches Used

### Deep Learning (CNN-Based)

#### Regression Model
- **Backbone**: EfficientNet (pretrained)
- **Loss Function**: Mean Absolute Error (L1 Loss)
- **Optimizer**: AdamW / SAM (Sharpness-Aware Minimization)
- **Output**: Continuous bone age (years)

#### Classification Model
- **Backbone**: EfficientNet (classification variant)
- **Loss Function**: Cross-Entropy Loss
- **Output**: 5 bone-age categories
- **Evaluation**: Accuracy, Precision, Recall, F1, QWK

---

### Classical Machine Learning Baseline

#### HOG + XGBoost
- **Feature Extraction**: Histogram of Oriented Gradients (HOG)
- **Model**: XGBoost
- **Purpose**: Baseline comparison against CNN models

---

## Evaluation Metrics

### Regression Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

### Classification Metrics
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**
- **Quadratic Weighted Kappa (QWK)**

QWK is used to evaluate ordinal agreement between predicted and true age categories.

---

## Model Explainability – Grad-CAM

Grad-CAM heatmaps are used to visualize regions of the hand X-ray that most influence predictions.

- **Red / Yellow regions** → High importance (growth plates, joints)
- **Blue regions** → Low importance (background)

This improves interpretability and clinical trust.

---

## Bias & Error Analysis

- Performance analyzed separately for **Male** and **Female** samples
- Error trends studied across age ranges
- Helps identify potential dataset imbalance or bias

---

## Results Summary

| Approach | Task | Observation |
|--------|------|-------------|
| EfficientNet (CNN) | Regression | MAE - 6.87 months, R² - 0.9498 |
| EfficientNet (CNN) | Classification | Accuracy - 77%, QWK - 0.9414, strong ordinal consistency |
| HOG + XGBoost | Both | Useful baseline, lower performance |

Deep learning models consistently outperform classical methods.

---

## Reproducibility

All experiments were conducted in **Kaggle notebooks**.

To reproduce results:
1. Open the corresponding notebook
2. Run all cells sequentially
3. Training instructions and configurations are documented in the notebooks

---

## References

- RSNA Bone Age Dataset
- Tan & Le, *EfficientNet*
- Selvaraju et al., *Grad-CAM*
- XGBoost Documentation

---

## Conclusion

This project presents a complete, interpretable, and reproducible pipeline for bone age prediction using hand radiographs. By combining deep learning, classical baselines, and explainability techniques, the system achieves strong predictive performance while remaining clinically meaningful.

---

If you find this project useful, feel free to star the repository!
