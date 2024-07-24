# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay

# Load dataset
df = pd.read_csv("heart_data.csv")

# Display dataset overview
print("First 5 rows of the data:")
print(df.head())
print("\nLast 5 rows of the data:")
print(df.tail())
print("Shape:")
print(df.shape)
print(df.info())
print("Statistical data:")
print(df.describe())
print(df['target'].value_counts())

# Prepare data for training
X = df.drop(columns='target', axis=1)
Y = df['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate model accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data:', test_data_accuracy)

# Example prediction
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print('Prediction:', prediction[0])

# Display prediction result
if (prediction[0] == 0):
    print('No Heart Disease')
else:
    print('Heart Disease')

# Visualize data distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Distribution of Target Variable')
plt.show()

# Plot histograms of features
df.hist(bins=15, figsize=(15, 10), layout=(4, 4), color='lightblue')
plt.tight_layout()
plt.show()

# Plot ROC curve
RocCurveDisplay.from_estimator(model, X_test, Y_test)
plt.title('ROC Curve')
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
