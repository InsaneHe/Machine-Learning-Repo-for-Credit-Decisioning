# This is the main.py file for using functions from other files named "models_xx.py"
# Package importing and warnings setting
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore') # Ignore specific warnings

# Data reading and data preprocessing
# Setting data path（我们可以考虑使用MySQL来导入，但在那之前先用接收用户输入的方式设定data_path）
''' Data path to use
data_path = \
    r"D:\PythonProjects\Github - Machine Learning Project for Credit Decisioning\Machine-Learning-Repo-for-Credit-Decisioning\Data\loan_data_2007_2014.csv"
'''

raw = input("Please enter the path of the data file: ").strip()
data_path = Path(raw.strip('"').strip("'")).resolve()   # Remove quotation marks and convert to absolute paths

if not data_path.is_file():
    exit("The file does not exist, the program terminates!\n")

suffix = data_path.suffix.lower()

try:
    if suffix == '.csv':
        df = pd.read_csv(data_path, encoding='utf-8-sig')
    else:
        df = pd.read_excel(data_path)
except Exception as e:
    exit(f"Attempt of reading data file failed: {e}")

print("Reading data file succeed!\nThe first 5 lines are:\n")
print(df.head())

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
    # Here, we can enter the model that we want to use
    model_map = {1: 'Logistic Regression',
                 2: 'Neural Network',
                 3: 'XGBoost'}
    prompt = ' / '.join(f'{k} {v}' for k, v in model_map.items())
    # Automatically generate prompt strings that consists of model_map elements

    while True:
        user_input = input(f"Please enter the number of the model you want to use:\n{prompt}\n(Enter 'quit' to quit): ").strip().lower()
        if user_input == "quit":
            print("Exit selected, bye!\n")
            break

        try:
            choice = int(user_input)
        except ValueError:
            print("Invalid input! Please enter a number or 'quit' to quit!\n")
            continue

        if choice not in model_map:
            print(f"You must enter a number between 1 and {len(model_map)}. Please try again!\n")
            continue

    model_to_use = model_map[choice]
    print("You have selected: ", model_to_use)

    # TODO: 待补充完善，选择模型后的后续措施（要调用其他文件里的模型）
    if choice == 1:  # After you choose "Logistic Regression"

    elif choice == 2:  # After you choose "Neural Network"

    elif choice == 3:  # After you choose "XGBoost"

    else:
        