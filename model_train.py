import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")



# Convert categorical values into numbers
data['gender'] = data['gender'].map({
    'female': 0,
    'male': 1
})

# Generate study_hours and sleep_hours manually
import numpy as np

np.random.seed(42)

data['study_hours'] = np.random.randint(1, 10, size=len(data))
data['sleep_hours'] = np.random.randint(4, 10, size=len(data))
data['attendance'] = np.random.randint(50, 100, size=len(data))

# Features
X = data[['study_hours', 'sleep_hours', 'attendance']]

# Target
y = data['math score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")