# Emotion Classification using SVM on Tweets

This project trains a Support Vector Machine (SVM) model to classify English-language tweets into six basic emotions using a pre-labeled dataset.

## 📁 Folder Structure

emotion-svm/
├── data/
│ └── emotions.csv # Input dataset (416k tweets)
├── models/
│ ├── svm_model_linear.pkl # Trained LinearSVC model
│ └── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── notebooks/
│ └── emotion_classifier.ipynb # Jupyter notebook (optional)
├── src/
│ └── train_model.py # Main training pipeline
├── results/
│ └── confusion_matrix.png # Sample confusion matrix
├── README.md
└── requirements.txt

shell
Copy
Edit
