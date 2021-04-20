# This is the model file where we run the implemented log reg model
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load data
train_feats = pd.read_csv('cleaned_feats/train_cleaned_features.csv')
dev_feats = pd.read_csv('cleaned_feats/dev_cleaned_features.csv')
test_feats = pd.read_csv('cleaned_feats/test_cleaned_features.csv')

train_feats.head(5)

# Clean data

# Format data for model

# LDA?

# Instantiate model - Multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Train model

# Test model (use scores)

# 