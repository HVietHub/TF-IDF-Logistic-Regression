import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\src")
from preprocessing import preprocess_text

def run_evaluation():
    model_path = Path(r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\artifacts_pi_vs_jb\model.joblib")
    test_csv_path = Path(r"C:\Users\FPT\Desktop\TF_IDF\test_unseen_pi_vs_jb.csv")
    output_report_path = Path(r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\reports_pi_vs_jb\test_unseen_detailed_report.json")

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test data: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    total_samples = len(df)
    print(f"Loaded {total_samples:,} samples.")
    print("Class distribution:\n", df["label"].value_counts().to_dict())

    # Preprocessing
    t0 = time.time()
    X_proc = df["prompt"].astype(str).apply(preprocess_text)
    prep_time = time.time() - t0

    # Prediction
    t1 = time.time()
    y_pred = model.predict(X_proc)
    y_proba = model.predict_proba(X_proc)
    infer_time = time.time() - t1

    y_true = df["label"].values
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro")
    macro_r = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    target_names = ["Prompt Injection (0)", "Jailbreak (1)"]
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    print("\n" + "=" * 70)
    print("             BÁO CÁO KIỂM THỬ MÔ HÌNH VỪA TRAIN")
    print("           Tập Dữ Liệu: test_unseen_pi_vs_jb.csv")
    print("=" * 70)
    print(f"Tổng số mẫu kiểm thử:     {total_samples:,}")
    print(f"Độ chính xác (Accuracy):  {acc * 100:.2f}%")
    print(f"Macro Precision:          {macro_p * 100:.2f}%")
    print(f"Macro Recall:             {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:           {macro_f1:.4f}")
    print(f"Weighted F1-Score:        {weighted_f1:.4f}")
    print("-" * 70)
    print(f"Thời gian tiền xử lý:     {prep_time:.3f}s ({prep_time / total_samples * 1000:.3f} ms / mẫu)")
    print(f"Thời gian suy luận:       {infer_time:.3f}s ({infer_time / total_samples * 1000:.3f} ms / mẫu)")
    print(f"Tốc độ xử lý:             {total_samples / infer_time:.1f} mẫu / giây")
    print("-" * 70)

    print("\nChi tiết từng nhãn:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Ma trận nhầm lẫn (Số lượng):")
    print(f"                              Dự đoán PI (0)    Dự đoán JB (1)")
    print(f"Thực tế Prompt Injection (0):        {cm[0, 0]:>7,}          {cm[0, 1]:>7,}")
    print(f"Thực tế Jailbreak (1):               {cm[1, 0]:>7,}          {cm[1, 1]:>7,}")

    print("\nMa trận nhầm lẫn (Tỷ lệ %):")
    print(f"Thực tế Prompt Injection (0):        {cm_norm[0, 0]:>6.2f}%          {cm_norm[0, 1]:>6.2f}%")
    print(f"Thực tế Jailbreak (1):               {cm_norm[1, 0]:>6.2f}%          {cm_norm[1, 1]:>6.2f}%")
    print("=" * 70)

    # Attach prediction to df for inspection
    df["pred"] = y_pred
    df["prob_pi"] = y_proba[:, 0]
    df["prob_jb"] = y_proba[:, 1]
    df["is_correct"] = (df["label"] == df["pred"])

    # Length analysis
    df["length"] = df["prompt"].astype(str).str.len()
    short_df = df[df["length"] < 200]
    medium_df = df[(df["length"] >= 200) & (df["length"] < 1000)]
    long_df = df[df["length"] >= 1000]

    print("\nĐộ chính xác theo độ dài câu lệnh:")
    print(f"  - Ngắn (< 200 ký tự, {len(short_df)} mẫu):        {accuracy_score(short_df['label'], short_df['pred']) * 100:.2f}%")
    print(f"  - Vừa (200-1000 ký tự, {len(medium_df)} mẫu):     {accuracy_score(medium_df['label'], medium_df['pred']) * 100:.2f}%")
    print(f"  - Dài (>= 1000 ký tự, {len(long_df)} mẫu):       {accuracy_score(long_df['label'], long_df['pred']) * 100:.2f}%")

    # Save detailed JSON report
    report_data = {
        "dataset": str(test_csv_path),
        "total_samples": total_samples,
        "accuracy": round(acc, 5),
        "macro_precision": round(macro_p, 5),
        "macro_recall": round(macro_r, 5),
        "macro_f1": round(macro_f1, 5),
        "weighted_f1": round(weighted_f1, 5),
        "inference_time_seconds": round(infer_time, 4),
        "inference_per_sample_ms": round(infer_time / total_samples * 1000, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_pct": cm_norm.tolist(),
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Đã lưu báo cáo chi tiết vào: {output_report_path}")

if __name__ == "__main__":
    run_evaluation()
