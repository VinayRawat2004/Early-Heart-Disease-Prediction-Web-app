import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("heart_data.csv")

# Prepare data for training
X = df.drop(columns='target', axis=1)
Y = df['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as model.pkl")
