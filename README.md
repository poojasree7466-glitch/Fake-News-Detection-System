# 📊 Fake News Detection System & Text Analytics

> An end-to-end Machine Learning and Natural Language Processing (NLP) project that predicts whether a given news article is **Real** or **Fake** using a Logistic Regression Classifier and extracts semantic text patterns through rigorous feature optimization.

---

## 📌 Project Overview
The proliferation of misinformation across digital media platforms has made content verification an essential challenge. Distinguishing between authentic journalism and fabricated narratives manually is highly resource-intensive, making automated linguistic analysis an essential modern solution.

This project combines **Supervised Machine Learning** and **Advanced Text Mining** techniques to:
* Predict whether a newly inputted news block is trustworthy or fabricated.
* Parse structural differences in language layout between real data and disinformation.
* Generate meaningful textual insights that support automated fact-checking pipelines.

---

## 🎯 Objectives
The primary objectives of this project are:
* Perform comprehensive **Exploratory Data Analysis (EDA)** on textual structures.
* Clean and preprocess raw unformatted text corpora using tokenization frameworks.
* Design a optimized feature pipeline using statistical word weight matrices.
* Build a machine learning classification model optimized for high-dimensional text data.
* Evaluate generalization parameters using precise matrix profiles (Accuracy, Precision, Recall).
* Deploy an interactive, user-facing interface for real-time inference checks.

---

## 📂 Dataset
The system operates on structural news corpora comprising two distinct collection matrices:
* `True.csv`: Contains verified articles from official trusted news agencies.
* `Fake.csv`: Contains flagged misinformation and unverified digital content.

The text fields analyzed include:
* **Title**: Headline context of the article.
* **Text**: Full body text containing the core semantic parameters.
* **Subject**: The topical thematic area (e.g., worldnews, politics).
* **Date**: Temporal markers of publication.

---

## 📚 Project Workflow
Raw Dataset | ▼ Exploratory Data Analysis | ▼ Data Cleaning | ▼ Feature Engineering | ▼ Text Stemming & Normalization | ▼ Train-Test Split | ▼ Logistic Regression Model | ▼ Performance Evaluation | ▼ Model Serialization | ▼ Streamlit Web UI Deployment

---

## 🔍 Step 1: Exploratory Data Analysis (EDA)

Before building the model, extensive exploratory data analysis was performed to understand document structure, class imbalances, and text patterns.

### 📊 Data Inspection
The following initial checks were systematically executed:
* Dataset structural shape and dimension metrics.
* Feature data types profiling.
* Identification of missing or corrupted values.
* Elimination of duplicated records.
* Statistical class imbalance and target label distribution profiles.

```python
# Baseline Profiling Implementation
df.info()
df.describe()
df.isnull().sum()
```
###📈 Distribution Analysis

Histograms and kernel density plots were plotted for text metrics, specifically measuring total word counts and string character distributions across the real and fake categories.
📦 Box Plot Analysis

Box plots were used to compare total string lengths between real and fake documents to capture underlying structural signals:
Insights: Fake news content frequently exhibits a vastly wider variance in character lengths, showcasing either heavily bloated or extremely short paragraph patterns compared to real journalistic standards.
📊 Count Plot Analysis

Categorical variables were analyzed using count plots to assess thematic distributions:
Features Analyzed: Subject vs. Target Class
Insights: Disinformation patterns correlate strongly with specific structural category tags, emphasizing hyper-partisan political topics over highly standardized international wires.
🧹 Step 2: Data Cleaning & Preprocessing

Proper lexical cleaning is vital to avoid injecting structural noise into the high-dimensional feature matrix.
🔤 String Normalization & Symbol Removal

All non-alphabetic elements (numbers, punctuation marks, emojis) are systematically stripped out using customized regex pipelines.
cleaned_content = re.sub('[^a-zA-Z]', ' ', content).lower()
Stopwords Elimination
High-frequency syntactic filler words (such as 'the', 'is', 'on', 'which') are stripped utilizing the standard NLTK dictionary baseline.
🪵 Porter Stemming Normalization
Words are chopped back down to their linguistic core root tokens (e.g., "analyzing", "analyzed", "analyzes" all compress uniformly to "analyz").
⚙️ Step 3: Feature Engineering & Vectorization
🔢 TF-IDF Vectorization
Text arrays are mapped into structural numerical matrices using Term Frequency-Inverse Document Frequency matrix metrics, capturing term importance while depressing globally repetitive words.
✂️ Step 4: Train-Test Split
The feature matrix was split dynamically to ensure unbiased cross-evaluation profiles:
80% Training Matrix: Used exclusively for structural weight learning.
20% Testing Matrix: Kept isolated to validate true production performance.
🧠 Step 5: Machine Learning Training & Optimization
A fine-tuned Logistic Regression classifier was trained over the high-dimensional text vectors. Given the sparse nature of text arrays, it provides stable probability boundaries without overfitting risk.

🛠️ Technologies Used
Category                       Tools/Libraries
Programming Language           Python
Data Processing                Pandas, NumPy
Linguistic Processing          NLTK (Natural Language Toolkit)
Machine Learning               Scikit-learn
Model Deployed                 Logistic Regression
Web UI Engine                  Streamlit


👩‍💻 Author
Poojasri Kasani
If you found this project configuration helpful, feel free to ⭐ the repository!
