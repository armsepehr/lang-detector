import pickle

# Load the trained model
with open("language_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_language(text):
    """Predicts the language of a given text."""
    return model.predict([text])[0]

if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        print(f"Predicted language: {predict_language(text)}")
