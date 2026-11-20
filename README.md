# SMS Spam Classification using Machine Learning

## 1. Project Overview

This project uses Machine Learning to classify SMS messages into two categories:

- **Ham:** Normal, legitimate messages.
- **Spam:** Unwanted or promotional messages.

The goal is to build a model that can identify spam messages accurately using Natural Language Processing (NLP) and Scikit-learn.

## 2. Dataset

- Dataset: SMS Spam Collection
- Total messages after removing duplicates: 5,169
- Ham messages: 4,516
- Spam messages: 653

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- TF-IDF Vectorization

## 4. Project Workflow

1. Load the dataset.
2. Explore and understand the data.
3. Remove duplicate messages.
4. Separate features and target labels.
5. Split the dataset into training and testing sets.
6. Convert text into numerical features using TF-IDF.
7. Train three Machine Learning models.
8. Evaluate and compare model performance.
9. Test the best model on new SMS messages.
10. Save and reload the trained model.

## 5. Machine Learning Models

Three models were trained and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

## 6. Model Performance

The following table shows the performance for the Spam class.

| Model | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Naive Bayes | 1.000 | 0.626 | 0.770 |
| Logistic Regression | 0.961 | 0.748 | 0.841 |
| Linear SVM | 0.967 | 0.885 | 0.924 |

**Best Model: Linear SVM**

Linear SVM achieved the highest Spam recall and F1-score among the three tested models.

- Accuracy: 98.16%
- Spam Precision: 96.67%
- Spam Recall: 88.55%
- Spam F1-Score: 92.43%

## 7. Model Testing

The trained Linear SVM model was tested on new SMS messages.

Example predictions:

- "Hey, are we meeting tomorrow at 10 AM?" → Ham
- "Congratulations! You won a free prize. Claim your cash reward now!" → Spam

Both example messages were classified correctly.

## 8. Saving the Model

The trained model and TF-IDF vectorizer were saved using Joblib.

- `sms_spam_model.pkl`
- `tfidf_vectorizer.pkl`

Both files were successfully loaded and tested again without retraining.

## 9. Conclusion

This project demonstrates how Machine Learning and NLP can be used for SMS spam detection.

After comparing three models, Linear SVM performed best on the held-out test dataset based on Spam recall and F1-score.

## 10. Future Improvements

- Perform hyperparameter tuning.
- Test on a larger and more diverse dataset.
- Improve spam recall while maintaining good precision.
- Build a simple web application for real-time SMS classification.
