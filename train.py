import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from dataset import load_dataset
from model import get_model

# Load dataset
df = load_dataset()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["language"], test_size=0.3, random_state=42)

# Select a model (change to "naive_bayes" or "svm" or "logistic_regression" to test different algorithms)
model = get_model("logistic_regression")

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")  # Use "macro" or "micro" based on your requirement
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

print(f"Model Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Print detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("language_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'language_model.pkl'")
