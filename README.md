# Emotion_detection

Requirements:
1. Dataset
   Emotions dataset from kaggle
   link: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

2. Approach:
   1. Dataset
      Source: text_emotion.csv

      Only four emotions were selected: happy, sadness, anger, neutral.

  2. Text Preprocessing
    Text column (content) was used as input.
    No deep cleaning like removing stopwords or punctuation was performed (could be added for improvement).
    Data was split into X (text) and y (emotion labels).

  3. Feature Extraction
    Used TF-IDF Vectorization via TfidfVectorizer(max_features=5000):
    Converts text into numerical feature vectors.
    Captures importance of words in context of all documents.

  5. Model
    Logistic Regression:
    A linear model suitable for multiclass classification using one-vs-rest strategy.
    Trained using model.fit(X_train, y_train).

  6. Evaluation
    Accuracy: Checked how often the model predicted correctly.
    Confusion Matrix: Showed breakdown of correct and incorrect predictions across all emotion classes.
    Classification Report: Provided precision, recall, F1-score per class.

  7. Deployment Preparation
    Saved the model and vectorizer using pickle:
   A simple Streamlit UI allowed user input to be transformed and predicted using the saved model.


Dependencies:
  pandas
  numpy
  scikit-learn
  streamlit
