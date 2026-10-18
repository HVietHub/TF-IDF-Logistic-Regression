import joblib
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
sys.path.insert(0, r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\src")
from preprocessing import preprocess_text

# Load model
model_path = r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\artifacts_pi_vs_jb\model.joblib"
model = joblib.load(model_path)
print(f"Loaded model from: {model_path}")

# Load test data
test_path = r"C:\Users\FPT\Desktop\TF_IDF\test_unseen_pi_vs_jb.csv"
df = pd.read_csv(test_path)
print(f"Loaded test dataset: {len(df):,} samples")
print("Label distribution:\n", df["label"].value_counts().to_dict())

# Preprocess & Predict
t0 = time.time()
X_proc = df["prompt"].astype(str).apply(preprocess_text)
prep_time = time.time() - t0

t1 = time.time()
y_pred = model.predict(X_proc)
y_proba = model.predict_proba(X_proc)
infer_time = time.time() - t1

y_true = df["label"].values
acc = accuracy_score(y_true, y_pred)

print("\n" + "=" * 65)
print("              DETAILED TEST RESULTS SUMMARY")
print("=" * 65)
print(f"Total Samples:           {len(df):,}")
print(f"Overall Accuracy:        {acc*100:.2f}%")
print(f"Preprocessing Time:      {prep_time:.3f}s ({prep_time/len(df)*1000:.3f} ms/sample)")
print(f"Model Inference Time:    {infer_time:.3f}s ({infer_time/len(df)*1000:.3f} ms/sample)")
print("-" * 65)

print("\nClassification Report:")
target_names = ["Prompt Injection (0)", "Jailbreak (1)"]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix (Counts):")
print("                  Pred PI (0)   Pred JB (1)")
print(f"Actual PI (0):        {cm[0,0]:>7,}       {cm[0,1]:>7,}")
print(f"Actual JB (1):        {cm[1,0]:>7,}       {cm[1,1]:>7,}")

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
print("\nConfusion Matrix (Normalized %):")
print(f"Actual PI (0):        {cm_pct[0,0]:>6.2f}%       {cm_pct[0,1]:>6.2f}%")
print(f"Actual JB (1):        {cm_pct[1,0]:>6.2f}%       {cm_pct[1,1]:>6.2f}%")
print("=" * 65)

# Inspect errors
df["pred"] = y_pred
df["prob_pi"] = y_proba[:, 0]
df["prob_jb"] = y_proba[:, 1]
errors_pi_as_jb = df[(df["label"] == 0) & (df["pred"] == 1)]
errors_jb_as_pi = df[(df["label"] == 1) & (df["pred"] == 0)]

print(f"\nSample False Positives for JB (Actual PI misclassified as JB, count={len(errors_pi_as_jb)}):")
for i, r in errors_pi_as_jb.head(3).iterrows():
    prompt_snippet = str(r["prompt"])[:100].replace("\n", " ")
    print(f"- [Actual PI -> Pred JB ({r['prob_jb']*100:.1f}%)]: {prompt_snippet}...")

print(f"\nSample False Negatives for JB (Actual JB misclassified as PI, count={len(errors_jb_as_pi)}):")
for i, r in errors_jb_as_pi.head(3).iterrows():
    prompt_snippet = str(r["prompt"])[:100].replace("\n", " ")
    print(f"- [Actual JB -> Pred PI ({r['prob_pi']*100:.1f}%)]: {prompt_snippet}...")
