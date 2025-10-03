# This is the file for using functions from other files named as "models_xx.py"
# Package importing and warnings setting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore') # Ignore specific warnings

# Data reading and data preprocessing
data = pd.read_csv("")  # Here we need to fill in the path of the csv file we want to read

# Partitioning the dataset and training set
np.random.seed(123)  # Using random seed 123

test_size = float(input("Please enter the number to set the size of testing set: "))  # Set the test_size
train_size = float(input("Please enter the number to set the size of training set: "))  # Set the train_size
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=123)



