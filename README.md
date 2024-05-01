
<div align="center">
  <img src="src/assets/reliability.png" alt="Model Calibration Curves" width="48%">
  <img src="src/assets/risk_curve.png" alt="Reliability Diagram" width="48%">
</div>

# Churn Prediction System

## Introduction
The Churn Prediction System is designed to help businesses reduce customer attrition by identifying customers likely to churn. This system uses machine learning models trained on historical data to predict churn probability weekly. The operators can then proactively engage these customers to improve retention rates.

## Features
- **Model Calibration:** Ensures that the predicted probabilities of churn are realistic and actionable.
- **Weekly Prediction Schedule:** Automatically selects N customers each week who are most likely to churn, based on the model's predictions.
- **Operator Efficiency:** Optimizes the number of calls operators can make in a given week, maximizing the impact of customer interactions.

## Quick Start
### Prerequisites
Ensure you have Python 3.8 or higher installed on your system. You can download it from [Python's official site](https://www.python.org/downloads/).

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://your-repository-link.com
   cd your-repository-directory
   ```

2. **Setup a virtual environment** (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

4. **Install the project package:**
   ```bash
   pip install -e .
   ```

### Usage
To run the churn prediction model, execute the following command:
```bash
python run_churn_prediction.py
```

## Documentation
- **BaseTrainer:** Handles the training of the churn prediction model.
- **BasePredictor:** Responsible for making weekly predictions using the trained model.
- **Data Cleaning:** Scripts for preprocessing the data, handling missing values, and preparing the dataset for training.

For detailed documentation, visit [Documentation](https://your-documentation-link.com).

## Contributing
Contributions are welcome! Please read our [Contributing Guide](https://your-repository-link.com/CONTRIBUTING.md) for details on how to submit changes and for our code of conduct.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Project Link: [https://github.com/your-repository-link](https://github.com/your-repository-link)
