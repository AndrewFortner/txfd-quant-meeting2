import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Read minute resolution data for SPX from the 'SPX-M.txt' file
data = pd.read_csv('data/SPX-M.txt', sep=' ', header=None)

# Data preprocessing
data.columns = ['date', 'time', 'close']
# Convert date and time columns to datetime format
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time
data['close'] = data['close'].astype(float)
# compress [time, close] (n x 2 matrix) into a numpy array (1 x n) of close prices
data = data.groupby('date').apply(lambda x: x.set_index('time')['close'].to_numpy())

# Normalize our data by turning them into a list of deltas, i.e. the difference between each price and the previous price
data = data.apply(lambda x: np.diff(x))

# 1: the value at index j is greater than the value at index i
# 0: the value at index j is less than the value at index i
# define i, j to be whatever you like. For this example, we will use the entire range
i = 0
j = 250
truncated_data = data.apply(lambda x: x[i:j])
prediction_time = j+1
# we cannot predict a value in the past, so prediction_time must be greater than j
assert(prediction_time > j)
assert(i < j)

# Define ground truth labels
labels = data.apply(lambda x: 1 if x[prediction_time] > x[j] else 0)

# Convert data and labels into numpy arrays
label_values = labels.values

truncated_prices = np.vstack(truncated_data.values)

X_train, X_test, y_train, y_test = train_test_split(truncated_prices, label_values, test_size=0.2, random_state=42)

# Build an XGBoost classifier.
model = xgb.XGBClassifier(
    n_estimators=100,  # Adjust the number of trees as needed
    max_depth=3,       # Adjust the maximum depth of each tree as needed
    learning_rate=0.1  # Adjust the learning rate as needed
)

# Train the model on the training data.
model.fit(X_train, y_train)

# Make predictions on the testing data.
y_pred = model.predict(X_test)

# Evaluate the model's performance.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# You can also print a classification report for more detailed metrics.
print(classification_report(y_test, y_pred))