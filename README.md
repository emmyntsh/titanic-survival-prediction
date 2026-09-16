# Titanic Survival Prediction

An end-to-end machine learning project that explores passenger data from the Titanic dataset and develops classification models to predict passenger survival.

The project covers data cleaning, exploratory data analysis, feature engineering, model development, hyperparameter tuning, and leakage-aware model evaluation.

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on information such as passenger class, age, sex, fare, family relationships, cabin information, and ticket details.

Three machine learning approaches were explored:

- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP)

The models were first evaluated using baseline configurations before improving the Random Forest through hyperparameter tuning and developing a deeper MLP architecture.

## Dataset

This project uses the dataset from Kaggle's **Titanic - Machine Learning from Disaster** competition.

The dataset consists of:
- `train.csv` — 891 passenger records including the `Survived` target variable
- `test.csv` — 418 passenger records without the target variable

The dataset is not redistributed in this repository. It can be downloaded from the official Kaggle competition page:

**Dataset:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

After downloading the data, place `train.csv` and `test.csv` in the same directory as the notebook before running it.

## Project Workflow

The project follows the following workflow:

1. Data loading and understanding
2. Data quality and missing-value assessment
3. Data cleaning and preprocessing
4. Feature engineering and categorical encoding
5. Exploratory data analysis
6. Model preparation
7. Baseline model development
8. Leakage-safe data preparation
9. Random Forest hyperparameter tuning
10. MLP model improvement
11. Final model comparison and selection

Engineered features include passenger title, family size, travelling status, ticket group size, and cabin deck.

## Model Performance

Final validation performance using the leakage-safe evaluation workflow:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8268 | 0.7568 | 0.8116 | 0.7832 | 0.8719 |
| Tuned Random Forest | 0.8101 | 0.7612 | 0.7391 | 0.7500 | 0.8529 |
| Improved MLP | 0.8268 | 0.7714 | 0.7826 | 0.7770 | 0.8592 |

Logistic Regression was selected as the final model. Despite its relatively simple architecture, it achieved the highest recall, F1-score, and ROC-AUC among the evaluated models while remaining straightforward to interpret.

## Model Improvement

The Random Forest model was improved using GridSearchCV with 5-fold cross-validation. The selected configuration achieved an F1-score of 0.7500 on the held-out validation set.

The MLP was improved using a deeper neural network with multiple hidden layers, dropout regularization, class weighting, and early stopping. This increased its F1-score from 0.7287 in the baseline model to 0.7770.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Running the Project

1. Clone this repository.
2. Download `train.csv` and `test.csv` from the Kaggle Titanic competition.
3. Place both files in the same directory as `titanic_survival_prediction.ipynb`.
4. Open the notebook in Jupyter Notebook or Google Colab.
5. Run the notebook cells sequentially from top to bottom.

## Key Takeaways

This project reinforced the importance of evaluating the complete machine learning workflow rather than focusing only on model complexity. In particular, it highlighted the importance of preventing data leakage, using appropriate validation strategies, handling class imbalance, and comparing models using multiple evaluation metrics.

The final results also demonstrate that a simpler model can remain competitive with more complex approaches when the data is carefully prepared and evaluated.
