# Emotion Classification using SVM on Tweets

This project builds a **Support Vector Machine (SVM)** model to automatically classify English-language tweets into six basic emotions using a large, pre-labeled dataset. The objective is to identify the most suitable SVM kernel for the task and evaluate the model's performance in terms of accuracy and classification metrics.

---

##  Project Structure

emotionclassification/
├── data/
│ └── emotions.csv # Dataset with 416k labeled tweets
├── models/
│ ├── svm_model_linear.pkl # Trained SVM model with linear kernel
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer fitted on full data
├── src/
│ └── scrap_tweets.py # Script for training and clustering
├── results/
│ └── confusion_matrix.png # (Optional) Evaluation result visual
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sreyak03/SVM-emotionclasssifier.git
cd SVM-emotionclasssifier
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Training Script
bash
Copy
Edit
python src/scrap_tweets.py
This script:

Loads and preprocesses tweet data.

Compares multiple SVM kernels (Linear, RBF, Polynomial, Sigmoid).

Trains the best-performing model (Linear).

Saves the trained model and vectorizer.

Kernel Comparison
Kernel	Accuracy
Linear	89.89% 
RBF	86.52%
Polynomial	84.27%
Sigmoid	77.90%

Best Performing Kernel: Linear

The linear kernel showed the best performance, likely due to the sparse and high-dimensional nature of the TF-IDF features, which are well-suited to linear decision boundaries.

Final Model Performance
Tested on the full dataset (83,362 samples):

Accuracy: 89.89%

Precision (weighted avg): 0.90

Recall (weighted avg): 0.90

F1-score (weighted avg): 0.90

Per-Emotion Breakdown:
Emotion Label	Precision	Recall	F1-Score	Support
0 (Happy)	0.94	0.94	0.94	24238
1 (Sad)	0.92	0.93	0.92	28214
2 (Angry)	0.79	0.78	0.79	6911
3 (Fear)	0.90	0.91	0.90	11463
4 (Surprise)	0.85	0.84	0.84	9542
5 (Disgust)	0.73	0.72	0.73	2994

⚙ Model Saving
After training, the model and vectorizer are saved:

python
Copy
Edit
joblib.dump(final_model, 'models/svm_model_linear.pkl')
joblib.dump(vectorizer_full, 'models/tfidf_vectorizer.pkl')
These files can be loaded for emotion prediction without retraining.

 Emotion Prediction Example
python
Copy
Edit
import joblib

model = joblib.load("models/svm_model_linear.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = ["I am feeling very anxious and afraid."]
X = vectorizer.transform(text)
prediction = model.predict(X)
print("Predicted Emotion Label:", prediction)
 Future Enhancements
Add a Streamlit web app for real-time tweet emotion prediction.

Integrate an alerting system for flagged emotions (e.g., repeated anger or fear).

Explore unsupervised clustering for emotion-based groupings.

Incorporate deep learning models such as BiLSTM or BERT for comparison.
