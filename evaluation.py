import csv
from sklearn.metrics import *
from itertools import chain

if __name__ == "__main__":
    gold = input("Type the .csv gold label file (ex: gold.csv):")
    gold_file = gold
    pred = input("Type the .csv prediction file (ex: pred.csv):")
    pred_file = pred
    with open(gold_file, newline='') as gf:
        y_true = list(csv.reader(gf))
        y_true = list(chain.from_iterable(y_true))
        y_true = [float(i) for i in y_true]
    with open(pred_file, newline='') as pf:
        y_pred = list(csv.reader(pf))
        y_pred = list(chain.from_iterable(y_pred))
        y_pred = [float(i) for i in y_pred]
    # TODO: what metric should we use?
    try:
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
    except Exception as e:
        print("Invalid files")
        print("Exception: ", e)
