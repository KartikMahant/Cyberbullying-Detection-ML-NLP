<h1 align="center">🛡️ Cyberbullying Detection System</h1>
<p align="center">
  <b>NLP + Machine Learning Project for Detecting Cyberbullying in Tweets</b><br>
  <i>Built with TF-IDF, Logistic Regression & Streamlit</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-purple">
  <img src="https://img.shields.io/badge/Accuracy-82.82%25-green">
  <img src="https://img.shields.io/badge/F1%20Score-0.827-blue">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red">
</p>

---

<h2>🚀 Features</h2>

<ul>
  <li>🔍 Classifies tweets into <b>6 categories</b></li>
  <li>📊 Interactive Streamlit Dashboard</li>
  <li>🧠 TF-IDF + Machine Learning pipeline</li>
  <li>📈 EDA with charts & word clouds</li>
  <li>⚡ Real-time prediction with probabilities</li>
</ul>

---

<h2>🧠 Model Performance</h2>

<ul>
  <li><b>Best Model:</b> Logistic Regression</li>
  <li><b>Accuracy:</b> 82.82%</li>
  <li><b>F1 Score (Macro):</b> 0.827</li>
</ul>

<h3>⚙️ Hyperparameters</h3>

<pre>
C = 1.0
max_iter = 10000
solver = saga
</pre>

---

<h2>📂 Project Structure</h2>

<pre>
📦 cyberbullying-detector
 ┣ 📜 app.py
 ┣ 📓 cyberbullying_nlp.ipynb
 ┣ 📦 best_model.pkl
 ┣ 📦 tfidf_vectorizer.pkl
 ┣ 📦 label_encoder.pkl
 ┣ 📄 model_metadata.json
 ┣ 📊 new_tweets.csv
 ┣ 📄 requirements.txt
 ┗ 📄 .gitignore
</pre>

---

<h2>🧹 Text Preprocessing</h2>

<ul>
  <li>Lowercasing text</li>
  <li>Remove URLs, mentions, hashtags</li>
  <li>Remove punctuation & numbers</li>
  <li>Stopword removal (keeping negations)</li>
  <li>Lemmatization (NLTK)</li>
</ul>

---

<h2>⚙️ Feature Engineering</h2>

<ul>
  <li>TF-IDF Vectorizer</li>
  <li>Max Features: 30,000</li>
  <li>N-grams: (1,2)</li>
  <li>Min DF: 2 | Max DF: 0.95</li>
</ul>

---

<h2>⚖️ Handling Imbalance</h2>

<ul>
  <li>SMOTE (Oversampling)</li>
  <li>class_weight = balanced</li>
  <li>Stratified K-Fold Cross Validation</li>
</ul>

---

<h2>📊 Exploratory Data Analysis</h2>

<ul>
  <li>Class Distribution</li>
  <li>Tweet Length Analysis</li>
  <li>Word Clouds</li>
  <li>Top Words per Class</li>
  <li>Correlation Heatmaps</li>
</ul>

---

<h2>🔍 Run the Project</h2>

<pre>
git clone https://github.com/your-username/cyberbullying-detector.git
cd cyberbullying-detector
pip install -r requirements.txt
streamlit run app.py
</pre>

---

<h2>🧪 Example</h2>

<pre>
Input:
"You are so dumb and useless"

Output:
✔ Predicted Class
✔ Confidence Scores
✔ Cleaned Text
</pre>

---

<h2>📦 Dependencies</h2>

<p>
streamlit • numpy • pandas • matplotlib • seaborn • scikit-learn • nltk • wordcloud • imbalanced-learn
</p>

---

<h2>📌 Future Improvements</h2>

<ul>
  <li>🤖 Deep Learning (BERT / LSTM)</li>
  <li>🌍 Multilingual support</li>
  <li>🚀 API deployment</li>
  <li>🔍 Explainable AI (SHAP/LIME)</li>
</ul>

---

<h2>👨‍💻 Author</h2>

<p>Kartik Mahant</p>


---

</p>
