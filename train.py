import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import load_dataset
from model import get_model

# Load dataset
df = load_dataset()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["language"], test_size=0.3, random_state=42)

# Select a model (change to "svm" or "logistic_regression" to test different algorithms)
model = get_model("naive_bayes")

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
with open("language_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'language_model.pkl'")
