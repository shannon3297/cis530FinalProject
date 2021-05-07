import csv
from sklearn.metrics import *
from itertools import chain
import pandas as pd 

if __name__ == "__main__":
    # get file names 
    gold_file = input("Type the .csv gold label file (ex: gold.csv):")
    pred_file = input("Type the .csv prediction file (ex: pred.csv):")

    # read csvs
    y_true = pd.read_csv(gold_file)
    y_pred = pd.read_csv(pred_file)
    
    # get and print score 
    # labels = ['negative', 'neutral', 'positive']
    labels = [0, 1, 2]
    try:
        f1_score_classes = f1_score(y_true, y_pred, labels=labels, average=None)
        f1_score_overall = f1_score(y_true, y_pred, labels=labels, average='weighted')
        print("F1 score for each class: ", f1_score_classes)
        print("F1 score overall: ", f1_score_overall)
    except Exception as e:
        print("Invalid files")
        print("Exception: ", e)
