import os
import pickle
import fitz  # PyMuPDF
import numpy as np
import warnings
import pandas as pd

from pathlib import Path
from scipy.sparse import hstack

warnings.filterwarnings("ignore")
# --- CẤU HÌNH ĐƯỜNG DẪN CHUẨN ---
# Path(__file__) là file 'src/find.py'
# .parent là thư mục 'src'
# .parent.parent là thư mục gốc 'cv_job_classifier'
BASE_DIR = Path(__file__).resolve().parent.parent

# Trỏ thẳng vào thư mục model ở ngoài cùng (ngang hàng với src)
MODEL_DIR = BASE_DIR / "model"
INPUT_DIR = BASE_DIR / "input"

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        return text.strip()
    except Exception as e:
        print(f"Lỗi đọc file {pdf_path.name}: {e}")
        return None

def run_test():
    # 1. Load các file từ thư mục /model/
    try:
        with open(MODEL_DIR / "model.pkl", "rb") as f:
                model = pickle.load(f)
        with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)        
        # Nếu bạn có dùng scaler thì load thêm, không thì bỏ dòng dưới
        # with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        #     scaler = pickle.load(f)
        print(f"✅ Đã tải thành công model từ: {MODEL_DIR}")
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file model. Kiểm tra lại đường dẫn: {e}")
        return
    # 2. Nhập Job Description
    print("-" * 30)
    job_desc = input("Nhập mô tả công việc để test: ")
    print("-" * 30)

   # 3. Quét PDF
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text: continue

        # --- BƯỚC QUAN TRỌNG: TẠO 9005 FEATURES ---
        
        # A. Xử lý văn bản (9000 features)
        combined_text = f"{cv_text} {job_desc}"
        X_text_sparse = vectorizer.transform([combined_text])
        
        # B. Trích xuất 5 đặc trưng số (như trong train.py)
        # 1. num_skills (tạm tính bằng số từ trong cv_text chia cho một hệ số hoặc đếm dấu phẩy)
        num_skills = cv_text.count(',') + 1 
        # 2. text_length
        text_length = len(combined_text)
        # 3, 4, 5. has_python, has_java, has_sql
        has_python = 1 if "python" in combined_text.lower() else 0
        has_java = 1 if "java" in combined_text.lower() else 0
        has_sql = 1 if "sql" in combined_text.lower() else 0
        
        # C. Scale 5 cột số này
        num_data = pd.DataFrame([[num_skills, text_length, has_python, has_java, has_sql]], 
                        columns=["num_skills", "text_length", "has_python", "has_java", "has_sql"])
        X_num_scaled = scaler.transform(num_data)
        
        # D. Gộp lại thành 9005 cột (Dùng hstack như lúc train)
        from scipy.sparse import hstack
        X_final = hstack([X_text_sparse, X_num_scaled])

        # 4. Dự đoán
        probs = model.predict_proba(X_final)[0]
        match_idx = list(model.classes_).index(True)
        score = probs[match_idx] * 100
        
        print(f"{pdf_path.name:<25} | {score:>12.2f}% | {'Match' if score > 50 else 'No'}")

if __name__ == "__main__":
    run_test()