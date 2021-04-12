import csv
from sklearn.metrics import *
from itertools import chain

if __name__ == "__main__":
    gold = input("Type the .csv gold label for test file (ex: gold_test.csv):")
    gold_file = gold
    with open(gold_file, newline='') as gf:
        y_true = list(csv.reader(gf))
        y_true = [float(i) for id, i in y_true]
    n = len(y_true)
    mean = sum(y_true) / n
    y_pred = [mean] * n
    print(mean)
    mean_absolute_error = mean_absolute_error(y_pred, y_true)
    mean_squared_error = mean_squared_error(y_pred, y_true)
    max_error = max_error(y_pred,y_true)
    explained_variance_score=explained_variance_score(y_pred,y_true)
    r2_score = r2_score(y_pred,y_true)
    print("Mean Squared Error: ", mean_squared_error)
    print("Mean Absolute Error: ", mean_absolute_error)
    print("Max Error: ", max_error)
    print("Explained Variance Score: ", explained_variance_score)
    print("R2 Score: ", r2_score)
