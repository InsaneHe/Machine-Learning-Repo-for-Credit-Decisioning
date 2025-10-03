# Defining the parameter grid
print("Now you can give me 3 integers to set the number of estimators (trees) in this model: ")
n_estimators_a = int(input("Please enter the 1st estimator: "))
n_estimators_b = int(input("Please enter the 2nd estimator: "))
n_estimators_c = int(input("Please enter the 3rd estimator: "))

print("Now you can give me 3 integers to set the values of depths of trees in this model: ")
depths_a = int(input("Please enter the 1st depth: "))
depths_b = int(input("Please enter the 2nd depth: "))
depths_c = int(input("Please enter the 3rd depth: "))

print("Now you can give me 3 integers to set the values of learning rates in this model: ")
learning_rate_a = int(input("Please enter the 1st learning rate: "))
learning_rate_b = int(input("Please enter the 2nd learning rate: "))
learning_rate_c = int(input("Please enter the 3rd learning rate: "))

print("Now you can give me 3 integers to set the number of estimators (trees) in this model: ")
n_estimators_a = int(input("Please enter the 1st estimator: "))
n_estimators_b = int(input("Please enter the 2nd estimator: "))
n_estimators_c = int(input("Please enter the 3rd estimator: "))

param_grid = {
    "n_estimators": [n_estimators_a, n_estimators_b, n_estimators_c],  # 3 different numbers of trees to be used in xgboost model
    "max_depth": [depths_a, depths_b, depths_c],  # The depth of trees
    "learning_rate": [0.01, 0.1, 0.3],  # The learning rate
    "gamma": [0, 1, 5]  # The regularization parameters
}

# Parameter tuning


# Model training



# Model evaluation



#