# Defining the parameter grid
print("Now you can give me 3 integers to set the number of estimators (trees) in this model: ")
n_estimators_a = int(input("Please enter the 1st estimator: "))  # we use 100
n_estimators_b = int(input("Please enter the 2nd estimator: "))  # we use 200
n_estimators_c = int(input("Please enter the 3rd estimator: "))  # we use 300

print("Now you can give me 3 integers to set the values of depths of trees in this model: ")
depths_a = int(input("Please enter the 1st depth: "))  # we use 3
depths_b = int(input("Please enter the 2nd depth: "))  # we use 6
depths_c = int(input("Please enter the 3rd depth: "))  # we use 9

print("Now you can give me 3 integers to set the values of learning rates in this model: ")
learning_rates_a = int(input("Please enter the 1st learning rate: "))  # we use 0.01
learning_rates_b = int(input("Please enter the 2nd learning rate: "))  # we use 0.1
learning_rates_c = int(input("Please enter the 3rd learning rate: "))  # we use 0.3

print("Now you can give me 3 integers to set the number of estimators (trees) in this model: ")
gamma_a = int(input("Please enter the 1st estimator: "))  # we use 0
gamma_b = int(input("Please enter the 2nd estimator: "))  # we use 1
gamma_c = int(input("Please enter the 3rd estimator: "))  # we use 5

param_grid = {
    "n_estimators": [n_estimators_a, n_estimators_b, n_estimators_c],  # 3 different numbers of trees to be used in xgboost model
    "max_depth": [depths_a, depths_b, depths_c],  # The depth of trees
    "learning_rate": [learning_rates_a, learning_rates_b, learning_rates_c],  # The learning rate
    "gamma": [gamma_a, gamma_b, gamma_c]  # The regularization parameters
}

# Parameter tuning


# Model training



# Model evaluation



#