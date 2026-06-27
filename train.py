import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download essential NLTK data
nltk.download('stopwords')
ps = PorterStemmer()

# 1. Text Preprocessing Function
def stemming(content):
    cleaned_content = re.sub('[^a-zA-Z]', ' ', content)
    cleaned_content = cleaned_content.lower()
    cleaned_content = cleaned_content.split()
    cleaned_content = [ps.stem(word) for word in cleaned_content if not word in stopwords.words('english')]
    return ' '.join(cleaned_content)

print("--- ML Model Training Started ---")

# 2. Loading Datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Assigning Labels (1 for Fake, 0 for True)
fake_df['label'] = 1
true_df['label'] = 0

# Taking 1000 rows from each to make a balanced 2000 rows dataset
fake_df = fake_df.head(1000)
true_df = true_df.head(1000)

# Combining both datasets
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Handling missing values and combining title and text
df = df.fillna('')
df['content'] = df['title'] + ' ' + df['text']

print("Cleaning text data (This will take only 10-15 seconds)...")
df['content'] = df['content'].apply(stemming)

# 3. Separating features and target
X = df['content'].values
Y = df['label'].values

# 4. Feature Extraction (TF-IDF Vectorization)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 5. Splitting Training and Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# 6. Training Logistic Regression Model
print("Training the model...")
model = LogisticRegression()
model.fit(X_train, Y_train)

# 7. Checking Accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Training Complete! Model Accuracy: {test_data_accuracy * 100:.2f}%")

# 8. Saving Model and Vectorizer files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Success: 'model.pkl' and 'vectorizer.pkl' saved successfully!")