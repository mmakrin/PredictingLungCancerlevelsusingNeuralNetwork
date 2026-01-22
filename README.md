# Lung Cancer Severity Prediction using Neural Networks

This project focuses on predicting **lung cancer severity levels (Low, Medium, High)** using machine learning and deep learning models. It was developed as part of the course **Neural Networks & Deep Learning**.

The study explores data preprocessing, classical classifiers, and multiple neural network architectures (MLP), including hyperparameter tuning and performance evaluation.

---

## Dataset Overview

- **Samples:** 1,000 patients  
- **Features:** 23  
- **Target:** Lung cancer severity level (Low / Medium / High)

### Feature Categories

**Demographic**
- Age, Gender

**Environmental**
- Air Pollution, Dust Allergy, Occupational Hazards

**Health Risks**
- Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoking

**Symptoms**
- Chest Pain, Coughing Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring

---

## Data Preprocessing

- Removed irrelevant columns: `index`, `Patient ID`
- Binary encoding:
  - Smoking
  - Chest Pain
- Feature categorization:
  - Frequent Cold
  - Clubbing of Finger Nails (4 severity levels)
- Target encoding:
  - Low → 0, Medium → 1, High → 2
- Feature scaling:
  - Min-Max Scaling (for KNN & Nearest Centroid)
  - StandardScaler (for Neural Networks)
- Outlier detection using Z-score (no significant outliers found)
- Correlation analysis (no highly correlated features > 0.9)
- Class imbalance handled using **class weights**

---

## Models Implemented

### Classical Models
- K-Nearest Neighbors (k=1, k=3)
- Nearest Centroid
- Logistic Regression (baseline)

### Neural Networks (MLP)

#### Architectures tested:
- Shallow NN (1 hidden layer)
- Deep NN (2 hidden layers)
- Final optimized model (via GridSearchCV + KerasClassifier)

#### Final Model Architecture:
- Input layer: 23 features
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 64 neurons (ReLU)
- Dropout: 30%
- Output layer: 3 neurons (Softmax)

#### Training Parameters:
- Optimizer: Adam
- Batch size: 16
- Epochs: 50
- Loss: Sparse Categorical Crossentropy

---

## Performance Summary
| Model | Test Accuracy |
|-------|---------------|
| Logistic Regression | ~97% |
| KNN (k=1, k=3) | ~99–100% |
| Nearest Centroid | ~73–89% |
| Shallow Neural Network | ~100% |
| Deep Neural Network | ~100% |
| Final Optimized Model | ~100% |

Training and test accuracy/loss curves were monitored using custom TensorFlow callbacks.
| Model | Test Accuracy |
|-------|-------
