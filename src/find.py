import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer

# 1. LOAD DATA & PREP (Giữ nguyên logic để đảm bảo data khớp 100%)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Nhớ check lại đường dẫn này nếu file tune_params.py bạn để ở vị trí khác nhé
data_path = os.path.join(BASE_DIR, "data", "resume_data.csv") 

df = pd.read_csv(data_path)
df = df.fillna("")
df.columns = df.columns.str.replace('\ufeff', '').str.lower().str.strip()
if "responsibilities.1" in df.columns:
    df = df.drop(columns=["responsibilities.1"])

def count_skills(x):
    try:
        parsed = ast.literal_eval(x)
        return len(parsed) if isinstance(parsed, list) else len(str(x).split())
    except:
        return len(str(x).split(',')) if ',' in str(x) else len(str(x).split())

df["num_skills"] = df["skills"].apply(count_skills)
df["text"] = (df["skills"]*3 + " " + df["skills_required"]*2 + " " + 
              df["responsibilities"] + " " + df["positions"] + " " + 
              df["degree_names"] + " " + df["major_field_of_studies"] + " " + df["positions"])
df["text_length"] = df["text"].apply(len)

for skill in ["python", "java", "sql"]:
    df[f"has_{skill}"] = df["text"].apply(lambda x: 1 if skill in x.lower() else 0)

df["label"] = df["matched_score"] > 0.5
y = df["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(df["text"], y, test_size=0.2, random_state=42, stratify=y)
num_features = ["num_skills", "text_length", "has_python", "has_java", "has_sql"]
X_train_num, X_test_num = train_test_split(df[num_features], test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=9000, ngram_range=(1,2), min_df=2, max_df=0.9, stop_words="english", sublinear_tf=True)
X_train_text = vectorizer.fit_transform(X_train_text)
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_train = hstack([X_train_text, X_train_num])

# =====================================================================
# 2. BẮT ĐẦU DÒ TÌM THÔNG SỐ CHO LIGHTGBM
# =====================================================================
print("Đang chạy Tuning, vui lòng đợi (khoảng 1-3 phút)...")

# Mục tiêu: Tối ưu hóa F1-score cho lớp False
f1_false_scorer = make_scorer(f1_score, pos_label=False)

# Các thông số sẽ đem ra xào nấu ngẫu nhiên
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [-1, 10, 20, 30],
    'num_leaves': [31, 50, 80, 100],
    'min_child_samples': [10, 20, 30, 40]
}

lgbm_base = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_grid,
    n_iter=30,               # Thử ngẫu nhiên 30 trường hợp
    scoring=f1_false_scorer, 
    cv=3,                    
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\n" + "="*50)
print("🎯 TÌM THẤY THÔNG SỐ TỐT NHẤT:")
print("="*50)
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")
print("="*50)