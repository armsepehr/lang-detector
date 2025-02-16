from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_model(model_type="naive_bayes"):
    """
    Returns a text classification model with TF-IDF.
    
    Supported models:
    - "naive_bayes": Multinomial Naive Bayes
    - "svm": Support Vector Machine
    - "logistic_regression": Logistic Regression
    """
    tfidf = TfidfVectorizer()

    if model_type == "naive_bayes":
        model = make_pipeline(tfidf, MultinomialNB())
    elif model_type == "svm":
        model = make_pipeline(tfidf, SVC(kernel="linear"))
    elif model_type == "logistic_regression":
        model = make_pipeline(tfidf, LogisticRegression(max_iter=500))
    else:
        raise ValueError("Unsupported model type!")

    return model
