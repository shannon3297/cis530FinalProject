import csv
from sklearn.metrics import *
from itertools import chain
import pandas as pd

if __name__ == "__main__":
    gold = input("Type the .csv gold label for test file (ex: gold_test.csv):")
    gold_file = gold
    with open(gold_file, newline='') as gf:
        y_true = list(csv.reader(gf))
        y_true = [float(i) for id, i in y_true]
    n = len(y_true)
    mean = sum(y_true) / n
    y_pred = [mean] * n
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv(gold_file.split(".")[0] + '_pred.csv', index=False, header=False)
    print("Check out the predictions in " + gold_file.split(".")[0] + "_pred.csv")