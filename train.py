import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# load data
df = pd.read_csv("data/resume.csv")

X = df['resume_text']
y = df['category']

# chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# vector hóa
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# predict
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred, zero_division=0))

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

y_pred_nb = nb_model.predict(X_test_vec)

print("Naive Bayes:")
print(classification_report(y_test, y_pred_nb, zero_division=0))