# This is the main.py file for using functions from other files named "models_xx.py"
# Package importing and warnings setting
import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Ignore specific warnings
sns.set()
from sklearn.model_selection import train_test_split
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
data_path_3 = "loan_data.mysql"  # TODO: Waiting for creating mysql function
data_path_list = [data_path_1, data_path_2]

# Functions
def readingData():
    for i, data_path in enumerate(data_path_list):  # Print all the data paths available
        print(f"{i+1}: {data_path}")

    index_chosen = int(input("Please enter the index of the path you want to choose: ")) - 1
    data_path_chosen = data_path_list[index_chosen]
    print(f"You choose {data_path_chosen}!\n")

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

    chosen_path = Path(data_path_chosen)
    suffix = chosen_path.suffix.lower()

    if suffix != '.mysql':
        if not chosen_path.is_file():  # Check if the data path is a file
            exit("The file does not exist, the program terminates!\n")

    try:
        if suffix == '.csv':
            df = pd.read_csv(chosen_path, encoding='utf-8-sig')
        elif suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(chosen_path)
        elif suffix == '.mysql':  # TODO: This is for the future version of accepting a MySQL file
            from sqlalchemy import create_engine

            engine = create_engine(

            )
            sql = "SELECT * FROM tablexxx"
            df = pd.read_sql(sql, con = engine)
        else:
            exit("This program only accepts csv/xls/xlsx!\n")

        print("Reading data file succeed!\nThe first 5 lines are:\n")
        print(df.head())

    except Exception as e:
        exit(f"Attempt of reading data file failed: {e}\n")

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
        
