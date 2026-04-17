import os
import pickle
import pandas as pd
import numpy as np
import warnings
import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

#
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "resume_data.csv")

df = pd.read_csv(data_path)

#
df = df.fillna("")
df.columns = df.columns.str.replace('\ufeff', '')
df.columns = df.columns.str.lower().str.strip()

df = df.rename(columns={
    "experiencere_requirement": "experience_requirement"
})

if "responsibilities.1" in df.columns:
    df = df.drop(columns=["responsibilities.1"])

print("Clean xong:", df.shape)

#FEATURE ENGINEERING
def count_skills(x):
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return len(parsed)
        return len(str(x).split())
    except:
        if ',' in str(x):
            return len(str(x).split(','))
        return len(str(x).split())

df["num_skills"] = df["skills"].apply(count_skills)

df["text"] = (
    df["skills"]*3 + " " +
    df["skills_required"]*2 + " " +
    df["responsibilities"] + " " +
    df["positions"] + " " +
    df["degree_names"] + " " +
    df["major_field_of_studies"] + " " +
    df["positions"]
)

df["text_length"] = df["text"].apply(len)
important_skills = ["python", "java", "sql"]

for skill in important_skills:
    df[f"has_{skill}"] = df["text"].apply(
        lambda x: 1 if skill in x.lower() else 0
    )

#LABEL
df["label"] = df["matched_score"] > 0.5
y = df["label"]

#SPLIT
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_features = [
    "num_skills",
    "text_length",
    "has_python",
    "has_java",
    "has_sql"
]

X_train_num, X_test_num = train_test_split(
    df[num_features],
    test_size=0.2,
    random_state=42,
    stratify=y
)

#TF-IDF
vectorizer = TfidfVectorizer(
    max_features=9000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_train_text = vectorizer.fit_transform(X_train_text)
X_test_text = vectorizer.transform(X_test_text)

#SCALE numeric
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

#COMBINE
X_train = hstack([X_train_text, X_train_num])
X_test = hstack([X_test_text, X_test_num])

#MODEL
model = LGBMClassifier(
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1,
    verbose=-1,              #hide warning
    learning_rate=0.03,
    max_depth=20,
    min_child_samples=30,
    n_estimators=300,
    num_leaves=50
)
model.fit(X_train, y_train)

#EVALUATE
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#SAVE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)