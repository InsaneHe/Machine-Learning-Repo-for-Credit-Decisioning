# This is the main.py file for using functions from other files named "models_xx.py"
# Package importing and warnings setting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore') # Ignore specific warnings

# Data reading and data preprocessing
# Setting data path
data_path = \
    r"D:\PythonProjects\Github - Machine Learning Project for Credit Decisioning\Machine-Learning-Repo-for-Credit-Decisioning\Data\loan_data_2007_2014.csv"

# Read the data
data = pd.read_csv(data_path)  # Here we need to fill in the path of the csv file we want to read

# Partitioning the dataset and training set
np.random.seed(123)  # Using random seed 123

test_size = float(input("Please enter the number to set the size of testing set: "))  # Set the test_size
train_size = float(input("Please enter the number to set the size of training set: "))  # Set the train_size
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=123)

if __name__ == "__main__":
