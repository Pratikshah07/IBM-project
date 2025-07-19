# Traffic Congestion Prediction Project

## Overview
This project implements a machine learning solution for predicting traffic congestion levels in urban areas. Using various traffic-related features such as vehicle count, speed, location, and time of day, the system predicts whether an area will experience high congestion levels.

## Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA) with visualizations
- Two machine learning models:
  - Logistic Regression
  - Decision Tree Classifier
- Model performance evaluation and interpretation
- Visual representation of results using confusion matrices
- Decision tree visualization for interpretability

## Dataset
The project uses traffic data with the following features:
- Timestamp: Date and time of the observation
- Location: Area type (Downtown, Suburb, Highway, Arterial)
- Vehicle_Count: Number of vehicles in the area
- Vehicle_Speed: Average speed of vehicles
- Peak_Off_Peak: Whether the time is during peak hours or not
- Congestion_Level: Traffic congestion level (0-5)

## Requirements
All required dependencies are listed in `requirements.txt`. Main dependencies include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the main analysis script:
```bash
python main.py
```

The script will:
1. Load and preprocess the traffic data
2. Perform exploratory data analysis with visualizations
3. Train and evaluate two machine learning models
4. Generate performance metrics and visualizations

## Model Details

### Logistic Regression
- Used for binary classification of high congestion vs. low congestion
- Features are preprocessed using OneHotEncoder for categorical variables
- Provides interpretable coefficients showing feature importance

### Decision Tree
- Maximum depth of 5 to prevent overfitting
- Provides visual representation of decision rules
- Offers an alternative, non-linear approach to prediction

## Results
The analysis provides:
- Classification reports showing precision, recall, and F1-score
- Confusion matrices for both models
- Feature importance analysis
- Visualizations of data distributions and correlations

## Project Structure
```
.
├── README.md
├── requirements.txt
├── main.py
└── traffic_data.csv
```

## Visualizations
The project generates several visualizations:
- Distribution plots for key features
- Correlation matrix heatmap
- Confusion matrices for both models
- Decision tree structure

## Future Improvements
Potential enhancements could include:
- Implementation of more advanced models (Random Forest, XGBoost)
- Real-time prediction capabilities
- Integration with traffic monitoring systems
- Addition of weather and event data
- Cross-validation for more robust model evaluation
