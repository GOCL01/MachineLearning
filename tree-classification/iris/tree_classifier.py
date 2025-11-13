import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("iris.csv")

# Separate features and target
X = data.drop("Species", axis=1)  # assuming the target column is named 'species'
y = data["Species"]

# Encode target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set and print accuracy
y_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model to a file
joblib.dump(clf, "iris_decision_tree_model.pkl")
print("Model saved as 'iris_decision_tree_model.pkl'")

joblib.dump(label_encoder, "label_encoder.pkl")
