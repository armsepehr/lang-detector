# Language Identification using Machine Learning

This project implements a language identification system using the Tatoeba dataset. The system supports multiple machine learning models, including Naïve Bayes, Support Vector Machines (SVM), and Logistic Regression.

## Features

* **Automatic dataset downloading & extraction**
* **Supports multiple classifiers** (Naïve Bayes, SVM, Logistic Regression)
* **TF-IDF vectorization** for feature extraction
* **Easy-to-use training and inference scripts**

## Dataset

The dataset is sourced from the [Tatoeba Project](https://tatoeba.org/eng/downloads). It contains multilingual sentences labeled with their respective languages. We filter the dataset to include the following languages:

* English (`<span>eng</span>`)
* French (`<span>fra</span>`)
* Spanish (`<span>spa</span>`)
* German (`<span>deu</span>`)
* Italian (`<span>ita</span>`)

We use a subset of **5,000 sentences per language** to train the models.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/language-identifier.git
   cd language-identifier
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```
   python train.py
   ```

## Model Performance

We evaluate different machine learning models for language identification:

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Naïve Bayes (Multinomial)   | 92.3%    |
| Support Vector Machine (SVM) | 94.7%    |
| Logistic Regression          | 93.5%    |

## How It Works

### Training the Model

To train a language identification model:

```
python train.py --model naive_bayes  # Change to svm or logistic_regression
```

This will:

* Download and extract the dataset if not available
* Train the selected model using TF-IDF vectorization
* Save the trained model as `<span>language_model.pkl</span>`

### Predicting a Language

After training, you can test the model with:

```
python predict.py
```

You can then enter a sentence, and the script will predict its language.

## References

* **TF-IDF (Term Frequency-Inverse Document Frequency):** Used for text feature extraction. [Reference](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* **Naïve Bayes for Text Classification:** McCallum & Nigam (1998). [Reference]()
* **Support Vector Machines (SVM):** Cortes & Vapnik (1995). [Reference]()
* **Logistic Regression for NLP:** Kleinbaum et al. (2002). [Reference]()

## Future Improvements

* Implement a deep learning-based approach using LSTMs or Transformers
* Expand language support
* Optimize feature selection for better accuracy
