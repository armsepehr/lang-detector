# Language Identification using Machine Learning

This project implements a language identification system using the Tatoeba dataset. The system supports multiple machine learning models, including Naïve Bayes, Support Vector Machines (SVM), and Logistic Regression.

## Features

* **Automatic dataset downloading & extraction**
* **Supports multiple classifiers** (Naïve Bayes, SVM, Logistic Regression)
* **TF-IDF vectorization** for feature extraction

## Dataset

The dataset is sourced from the [Tatoeba Project](https://tatoeba.org/eng/downloads). It contains multilingual sentences labeled with their respective languages. We filter the dataset to include the following languages:

* English (`<span>eng</span>`)
* French (`<span>fra</span>`)
* Spanish (`<span>spa</span>`)
* German (`<span>deu</span>`)
* Italian (`<span>ita</span>`)
* Korean (`<span>kor</span>`)
* Italian (`<span>ara</span>`)
* Indian (`<span>ind</span>`)
* Chiense (`<span>cmn</span>`)

We use a subset of **5,000 sentences per language** to train the models.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-repo/language-identifier.git
   cd language-identifier
   python3 -m venv .venv
   source .venv/bin/activate
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

We evaluate different machine learning models for language identification with `sample_size=10,000` and `class_number=10`:

| Model                        | Accuracy | F1  | Recall | Precision |
| ---------------------------- | -------- | --- | ------ | --------- |
| Naïve Bayes (Multinomial)   | 89%      | 86% | 94%    | 89%       |
| Support Vector Machine (SVM) | 95%      | 95% | 96%    | 95%       |
| Logistic Regression          | 94%      | 94% | 96%    | 94%       |

## Performance Metrics: SVM

| Language          | Precision | Recall | F1-score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| **Arabic**  | 1.00      | 0.88   | 0.94     | 2985    |
| **Chinese** | 0.70      | 1.00   | 0.82     | 2993    |
| **English** | 1.00      | 0.98   | 0.99     | 2987    |
| **French**  | 0.99      | 0.97   | 0.98     | 3073    |
| **German**  | 1.00      | 0.99   | 0.99     | 3017    |
| **Indian**  | 0.99      | 0.96   | 0.98     | 2977    |
| **Italian** | 0.99      | 0.96   | 0.97     | 2979    |
| **Korean**  | 1.00      | 0.87   | 0.93     | 3003    |
| **Russian** | 1.00      | 0.94   | 0.97     | 3019    |
| **Spanish** | 0.98      | 0.96   | 0.97     | 2967    |

## Performance Metrics: Naive Bayes

| Language          | Precision | Recall | F1-score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| **Arabic**  | 1.00      | 0.96   | 0.98     | 2985    |
| **Chinese** | 1.00      | 0.02   | 0.04     | 2993    |
| **English** | 0.99      | 1.00   | 0.99     | 2987    |
| **French**  | 1.00      | 0.99   | 1.00     | 3073    |
| **German**  | 1.00      | 1.00   | 1.00     | 3017    |
| **Indian**  | 1.00      | 0.99   | 0.99     | 2977    |
| **Italian** | 0.99      | 0.98   | 0.99     | 2979    |
| **Korean**  | 1.00      | 0.93   | 0.97     | 3003    |
| **Russian** | 1.00      | 0.99   | 0.99     | 3019    |
| **Spanish** | 0.47      | 0.99   | 0.64     | 2967    |

## Performance Metrics: Logistic Regression

| Language          | Precision | Recall | F1-Score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| **Arabic**  | 1.00      | 0.88   | 0.93     | 2985    |
| **Chinese** | 0.65      | 1.00   | 0.79     | 2993    |
| **English** | 0.99      | 0.97   | 0.98     | 2987    |
| **French**  | 0.99      | 0.97   | 0.98     | 3073    |
| **German**  | 1.00      | 0.98   | 0.99     | 3017    |
| **Indian**  | 0.99      | 0.95   | 0.97     | 2977    |
| **Italian** | 0.99      | 0.95   | 0.97     | 2979    |
| **Korean**  | 1.00      | 0.83   | 0.91     | 3003    |
| **Russian** | 1.00      | 0.92   | 0.96     | 3019    |
| **Spanish** | 0.98      | 0.95   | 0.96     | 2967    |

# Analysis of Classifier Performance

* **SVM** achieves the highest overall accuracy (95%) and balanced F1, precision, and recall values.
* **Logistic Regression** performs very similarly to SVM but lags slightly in overall accuracy (94%). Most languages, including English, French, German, Indian, Italian, Russian, and Spanish, have high precision, recall, and F1-scores, indicating a well-calibrated model.
* **Naïve Bayes** shows noticeably lower overall performance (89%), which is largely influenced by severe performance drops in certain language classes. The classifier shows a dramatic drop with a recall of only 0.02 and an F1-score of 0.04—even though the precision is perfect (1.00) when it does predict Chinese. This indicates that the model almost never predicts Chinese, missing almost all true cases.

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

* Fine-tuning the model or adjusting class weights might help improve Chinese precision further.
* Revisit the feature representation and model assumptions for Naïve Bayes, especially for Chinese and Spanish. Consider data rebalancing or alternative preprocessing methods.
* verify the algorithm for more language class and more sample size
* Implement a deep learning-based approach using LSTMs or Transformers
* Expand language support
* Optimize feature selection for better accuracy
