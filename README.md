<div align="center">
  <img src="uim/assets/logo.png" alt="uim" width="400" height="auto">
</div>


# Uncertainty in Machine Learning Models

## Introduction
Machine learning models, despite their powerful capabilities, often come with inherent uncertainties. These uncertainties can arise due to various factors including, but not limited to, model assumptions, data quality, feature noise, and algorithmic limitations. Understanding and addressing these uncertainties is crucial for building robust and reliable predictive models, especially in critical applications like healthcare, finance, and autonomous driving.

## Features
- **Model Calibration:** Ensures that the predicted probabilities are realistic and actionable.
- **Operator Efficiency:** Optimizes the number of interactions operators can make in a given week, maximizing the impact of customer interactions.

## Quick Start

### Prerequisites
Ensure you have Python 3.10 or higher installed on your system. You can download it from [Python's official site](https://www.python.org/downloads/).

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/RRL-dev/uncertainty-in-models
   cd uncertainty-in-models
   ```

2. **Setup a virtual environment** (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   ```

3. **Install the project package:**
   ```bash
   pip install -e .
   ```

### Usage
To train the model, execute the following command:
```python
from uim.models import BaseTrainer
from uim.utils import DATASET_CFG, MODEL_CFG

BaseTrainer(model_cfg=MODEL_CFG, data_cfg=DATASET_CFG).fit()
```

To run the model, execute the following command:
```python
from uim.models import BasePredictor
from uim.utils import MODEL_CFG

predictor = BasePredictor(model_cfg=MODEL_CFG)
score = predictor.predict_proba(X=predictor.samples)
```

## Documentation
- **BaseTrainer:** Handles the training of the model.
- **BasePredictor:** Responsible for making predictions using the trained model.
- **BaseTransformer and DerivedFeaturesTransformer:** Scripts for preprocessing the data, handling missing values, and preparing the dataset for training.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Project Link: [https://github.com/RRL-dev/uncertainty-in-models](https://github.com/RRL-dev/uncertainty-in-models)

## Resources
- **Model Calibration Codebase from Apple:** [Apple ML Calibration](https://github.com/apple/ml-calibration/tree/main)
- **Temperature Scaling Resource:** [Temperature Scaling by Geoff Pleiss](https://github.com/gpleiss/temperature_scaling)

## Detailed Documentation
### EstimatorWithCalibration
This class integrates a classifier with a calibration mechanism using logistic regression to provide calibrated probability estimates.

<div align="center">
  <img src="uim/assets/reliability.png" alt="Reliability Diagram" width="400">
</div>

#### Methods
- \`__init__\`: Initializes the EstimatorWithCalibration with a specific classifier.
- \`fit\`: Fit the primary classifier and calibrate it using the provided training and calibration datasets.
- \`calibrate\`: Calibrates the classifier using logistic regression based on the probabilities of the initial classifier.
- \`predict\`: Predict class labels for the given samples using the calibrated model.
- \`predict_proba\`: Predict class probabilities for the given samples using the calibrated model.

### BasePredictor
This class encapsulates the prediction process using a trained model loaded from a pickle file.

#### Methods
- \`__init__\`: Initializes the BasePredictor with the path to the trained model pickle file.
- \`load_model\`: Loads the trained model from the pickle file.
- \`predict_proba\`: Makes predictions using the loaded model.

### BaseTrainer
This class encapsulates the training process of a model with configuration loaded from a YAML file.

#### Methods
- \`__init__\`: Initializes the BaseTrainer with a configuration path.
- \`fit\`: Trains the model using parameters specified in the YAML configuration file and saves the trained model.
- \`fit_predict\`: Fits the model and predicts on the test set.
- \`save_model\`: Saves the trained model to the specified path in the configuration file.

## Risk Score Explanation and Methodology

<div align="center">
  <img src="uim/assets/risk_curve.png" alt="Model Calibration Curves" width="600">
</div>

### Define Fuzzy Sets and Membership Functions
Based on the calibrated probabilities, define three fuzzy sets:
- **Low Risk**: \( P < 0.3 \)
- **Medium Risk**: \( 0.2 < P < 0.7 \)
- **High Risk**: \( P > 0.6 \)

#### Membership Functions:
- **Low Risk**: Linear decrease from 1 at \( P=0 \) to 0 at \( P=0.3 \).
- **Medium Risk**: Triangular shape, increasing from 0 at \( P=0.2 \) to 1 at \( P=0.45 \), then decreasing back to 0 at \( P=0.7 \).
- **High Risk**: Sigmoid function starting to increase significantly at \( P=0.6 \) and reaching 1 at \( P=0.75 \).

### Example Calculation of Membership Degrees
Suppose a customer has a calibrated probability of 0.55. The membership degrees would be calculated as:
- **Low Risk**: \`max(0, (0.3 - 0.55) / (0.3 - 0)) = 0\`
- **Medium Risk**: Triangular shape peaks at 0.45, so:
    - Increasing side: \`(0.55 - 0.2) / (0.45 - 0.2) = 1.75\` (not possible, hence it must be capped at 1)
    - Decreasing side: \`(0.7 - 0.55) / (0.7 - 0.45) = 0.75\`
    - The minimum of the two calculations: \`min(1, 0.75) = 0.75\`
- **High Risk**: Using a simple linear approximation for the sigmoid around 0.6, assume linear from 0.6 to 0.75:
    - \`(0.55 - 0.6) / (0.75 - 0.6) = -0.33\` (since it's below 0.6, it remains 0)

### Fuzzy Logic Decision Making
Based on the membership degrees, you could have rules like:
- If **High Risk > 0.5** then initiate customer retention protocol.
- If **Medium Risk > 0.5** then send promotional offers.
- If **Low Risk > 0.5** then maintain normal engagement.

```python
# Dummy Python import for risk score calculation

from uim.modules import BaseRiskScore

risk_score_calculator = BaseRiskScore()

single_risk = risk_score_calculator.fit(probabilities=0.55)
batch_risk = risk_score_calculator.fit(probabilities=np.array([0.2, 0.4, 0.6, 0.8]))
```