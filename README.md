# Weather Prediction Project

## Overview
This project predicts whether it will rain tomorrow based on historical weather data. Using machine learning models, the project preprocesses the data, selects features, and trains classifiers to make accurate predictions.

## Accesibility
This project can be viewed under [**this link**](https://serzeliuk.github.io/Weather-Prediction/) as well as a **Jupyter Notebook** ***WeatherPrediction.ipynb***

## Features
- **Data Cleaning**: Handles missing values and transforms categorical data into numerical values.
- **Feature Selection**: Uses correlation analysis to identify important features.
- **Modeling**: Implements Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, Voting Classifier, and a Neural Network.
- **Evaluation**: Compares models using accuracy, precision, recall, and F1-score.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `tensorflow`

Install dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```

## Workflow
1. **Data Preprocessing**: Convert categorical values, handle missing data.
2. **Feature Selection**: Identify the top features based on correlation.
3. **Model Training**: Train models on both the full dataset and selected important features.
4. **Evaluation**: Compare model performance across various metrics.
5. **Visualization**: Plot results for comparison.

## Models Implemented
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Voting Classifier
- Neural Network

## Results
![image](https://github.com/user-attachments/assets/68ac9684-9d15-421a-b562-047455f02780)

Model performance is evaluated using accuracy, precision, recall, and F1-score. Comparisons are made between the models trained on the full dataset and those trained on important features.

## License
This project is licensed under the MIT License.
