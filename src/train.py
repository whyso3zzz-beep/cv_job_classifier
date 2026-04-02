import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "resume_data.csv")

df = pd.read_csv(data_path)
# 1. fill NaN
df = df.fillna("")

# 2. rename cột lỗi
df = df.rename(columns={
    "educationaL_requirements": "educational_requirements",
    "experiencere_requirement": "experience_requirement"
})

# 3. drop cột thừa
if "responsibilities.1" in df.columns:
    df = df.drop(columns=["responsibilities.1"])

print("Clean xong:", df.shape)

df["text"] = (
    df["skills"] + " " +
    df["skills_required"] + " " +
    df["career_objective"] + " " +
    df["degree_names"] + " " +
    df["major_field_of_studies"] + " " +
    df["responsibilities"] + " " +
    df["positions"]
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english"
)
X = vectorizer.fit_transform(df["text"])

y = df["matched_score"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=200,max_depth=None,n_jobs=-1,random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("RandomForest score:", score)


os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)