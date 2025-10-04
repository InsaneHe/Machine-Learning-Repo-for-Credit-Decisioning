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
learning_rate_a = int(input("Please enter the 1st learning rate: "))  # we use 0.01
learning_rate_b = int(input("Please enter the 2nd learning rate: "))  # we use 0.1
learning_rate_c = int(input("Please enter the 3rd learning rate: "))  # we use 0.3

print("Now you can give me 3 integers to set the number of estimators (trees) in this model: ")
gamma_a = int(input("Please enter the 1st estimator: "))  # we use 0
gamma_b = int(input("Please enter the 2nd estimator: "))  # we use 1
gamma_c = int(input("Please enter the 3rd estimator: "))  # we use 5

param_grid = {
    "n_estimators": [n_estimators_a, n_estimators_b, n_estimators_c],  # 3 different numbers of trees to be used in xgboost model
    "max_depth": [depths_a, depths_b, depths_c],  # The depth of trees
    "learning_rate": [learning_rate_a, learning_rate_b, learning_rate_c],  # The learning rate
    "gamma": [gamma_a, gamma_b, gamma_c]  # The regularization parameters
}

# Parameter tuning
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=123)
# 5 threshold cross-validation, repeat 3 times (GridSearchCV has no repeats parameter, we can use RepeatedStratifiedKFold to achieve that) 此函数在main.py文件中

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=123
)

grid = GridSearchCV(  # 此函数在main.py文件中
    estimator=xgb_model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)  # 这两个参数在main.py中

print("The best combination of parameters is: ", grid.best_params_)  # print the best combination of parameters

# Establish the final model
final_model = xgb.XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",
    n_estimators=grid.best_params_["n_estimators"],
    max_depth=grid.best_params_["max_depth"],
    learning_rate=grid.best_params_["learning_rate"],
    gamma=grid.best_params_["gamma"],
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=123
)
final_model.fit(X_train, y_train)

# Performing predictions on the test set
pred_labels = final_model.predict(X_test)  # Return the category tag (0 or 1)

# Evaluate the model
print("The confusion matrix is: ")
print(confusion_matrix(y_test, pred_labels))  # 此函数在main.py文件中
print("\nClassification report is as below:\n")
print(classification_report(y_test, pred_labels))
