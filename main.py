# This is the main.py file for using functions from other files named "models_xx.py"
# Package importing and warnings setting
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore') # Ignore specific warnings
sns.set()
# TODO: 以下两行先注释掉，后续运行时可以解开注释以观察效果
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

# Data reading
# Setting data path（我们可以考虑使用MySQL来导入，但在那之前先用接收用户输入的方式设定data_path）
# Data path to use (data_path_1 == smaller dataset[3904*75], data_path_2 == bigger dataset[1048576*75])
data_path_1 = \
    r"D:\PythonProjects\Github - Machine Learning Project for Credit Decisioning\Machine-Learning-Repo-for-Credit-Decisioning\Data\loan_data_2007_2014.csv"
data_path_2 = \
    r"D:\PythonProjects\Github - Machine Learning Project for Credit Decisioning\Machine-Learning-Repo-for-Credit-Decisioning\Data\Bigger\loan_data_2007_2014_bigger.csv"
data_path_list = [data_path_1, data_path_2]

def readingData():
    for data_path in data_path_list:  # Print all the data paths available
        print(f"The index of the data path ({data_path}): ")
        print(f"{data_path.index()}\n")

    index_chosen = input("Please enter the index of the path you want to choose: ")
    print(f"You choose {data_path_list[index_chosen]}!\n")

    while True:
        cancel_or_not = int(input("Do you want to cancel?\n1).No\t2).Yes"))
        if cancel_or_not == 1:
            print("Proceed!\n")
            break
        elif cancel_or_not == 2:
            print("Cancelled!\n")
            return
        else:
            print("Please enter integer 1 (to PROCEED) or 2 (to CANCEL)!\n")
            continue

    if not data_path_list[index_chosen].is_file():  # Check if the data path is a file
        exit("The file does not exist, the program terminates!\n")

    suffix = data_path_list[index_chosen].suffix.lower()
    if suffix in ['.csv', '.xls', '.xlsx']:
        try:
            if suffix == '.csv':
                df = pd.read_csv(data_path, encoding='utf-8-sig')
            elif suffix == '.xls' or '.xlsx':
                df = pd.read_excel(data_path)

            print("Reading data file succeed!\nThe first 5 lines are:\n")
            print(df.head())
        except Exception as e:
            exit(f"Attempt of reading data file failed: {e}")
    else:
        print("This program only accepts csv/xls/xlsx!\n")


# Main function
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
        
