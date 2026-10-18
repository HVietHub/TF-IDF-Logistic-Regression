"""
Build Binary Dataset: Prompt Injection vs Jailbreak
===================================================
Creates a dataset strictly containing:
  - Label 0: Prompt Injection
  - Label 1: Jailbreak
Columns: ["prompt", "label"]

Excludes all benign prompts and guarantees zero leakage against test.csv.
"""

import os
import sys
import unicodedata
import re
import pandas as pd
import numpy as np

def normalize_text_key(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = unicodedata.normalize("NFC", text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def load_test_prompts(test_csv_path: str) -> set:
    if not os.path.exists(test_csv_path):
        return set()
    df_test = pd.read_csv(test_csv_path)
    col = "text" if "text" in df_test.columns else df_test.columns[0]
    test_keys = set(normalize_text_key(p) for p in df_test[col].dropna())
    print(f"Loaded {len(test_keys):,} test prompt keys to exclude from training.")
    return test_keys

def build_pi_vs_jb_dataset(
    dataset_dir: str = r"C:\Users\FPT\Desktop\TF_IDF\Dataset",
    test_csv_path: str = r"C:\Users\FPT\Desktop\TF_IDF\test.csv",
    output_path: str = r"C:\Users\FPT\Desktop\TF_IDF\Dataset\Datasets_main\pi_vs_jb_dataset.csv",
    secondary_output: str = r"C:\Users\FPT\Desktop\TF_IDF\Dataset\pi_vs_jb_dataset.csv",
    random_seed: int = 42,
):
    np.random.seed(random_seed)
    test_keys = load_test_prompts(test_csv_path)

    pi_records = []
    jb_records = []

    def add_item(prompt_str, label_int, source_name):
        p = str(prompt_str).strip()
        if not p or p.lower() in ("nan", "none") or len(p) < 5:
            return
        k = normalize_text_key(p)
        if k in test_keys:
            return
        item = {"prompt": p, "label": label_int, "norm_key": k, "source": source_name}
        if label_int == 0:
            pi_records.append(item)
        elif label_int == 1:
            jb_records.append(item)

    print("\n--- Collecting Prompt Injection samples (Label 0) ---")
    
    # 1. SPML Prompt Injection
    p_spml = os.path.join(dataset_dir, "Prompt_Injection", "spml_prompt_injection.csv")
    if os.path.exists(p_spml):
        df_spml = pd.read_csv(p_spml)
        for _, r in df_spml[df_spml["Prompt injection"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
            add_item(r["User Prompt"], 0, "spml")
        print(f"SPML: gathered {len(pi_records):,} PI samples.")

    # 2. Enterprise Prompt Injection
    p_ent = os.path.join(dataset_dir, "enterprise-prompt-injection-dataset", "enterprise_prompt_injection.csv")
    if os.path.exists(p_ent):
        df_ent = pd.read_csv(p_ent)
        c0 = len(pi_records)
        for _, r in df_ent[df_ent["label"].astype(str).str.strip().str.upper() == "INJECTION"].iterrows():
            add_item(r["text"], 0, "enterprise")
        print(f"Enterprise: added {len(pi_records) - c0:,} PI samples.")

    # 3. prompt-injections dataset
    p_pi = os.path.join(dataset_dir, "prompt-injections", "prompt_injections.csv")
    if os.path.exists(p_pi):
        df_pi = pd.read_csv(p_pi)
        c0 = len(pi_records)
        for _, r in df_pi[df_pi["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
            add_item(r["text"], 0, "prompt_injections")
        print(f"prompt-injections: added {len(pi_records) - c0:,} PI samples.")

    # 4. Datasets_main (root and subfolder) for any label 1
    for p_dm in [
        os.path.join(dataset_dir, "Datasets_main", "dataset.csv"),
        r"C:\Users\FPT\Desktop\TF_IDF\Datasets_main\dataset.csv",
    ]:
        if os.path.exists(p_dm):
            df_dm = pd.read_csv(p_dm, encoding="utf-8", encoding_errors="replace", low_memory=False)
            c0 = len(pi_records)
            for _, r in df_dm[df_dm["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
                add_item(r["prompt"], 0, "datasets_main_pi")
            print(f"datasets_main ({os.path.basename(p_dm)}): added {len(pi_records) - c0:,} PI samples.")

    # 5. Additional Prompt Injection from PIGuard train.json
    p_piguard = r"C:\Users\FPT\Desktop\IAP\pi-guard\workspaces\vietpmh\Task Completed\Task Meeting 5\PIGuard\datasets\train.json"
    if os.path.exists(p_piguard):
        import json
        with open(p_piguard, "r", encoding="utf-8") as fp:
            piguard_data = json.load(fp)
        c0 = len(pi_records)
        for item in piguard_data:
            if str(item.get("label", "")).strip() in ("1", "1.0"):
                src_name = f"piguard_{item.get('source', 'unknown')}"
                add_item(item.get("prompt", ""), 0, src_name)
        print(f"PIGuard train.json: added {len(pi_records) - c0:,} PI samples.")

    print(f"Total raw PI candidates: {len(pi_records):,}")

    print("\n--- Collecting Jailbreak samples (Label 1) ---")
    
    # 6. jailbreak_llms
    jb_base = os.path.join(dataset_dir, "jailbreak_llms", "prompts")
    p_jb1 = os.path.join(jb_base, "jailbreak_prompts_2023_05_07.csv")
    p_jb2 = os.path.join(jb_base, "jailbreak_prompts_2023_12_25.csv")
    if os.path.exists(p_jb1):
        df_jb1 = pd.read_csv(p_jb1, on_bad_lines="skip")
        for _, r in df_jb1.iterrows():
            add_item(r.iloc[0], 1, "jailbreak_llms_0507")
    if os.path.exists(p_jb2):
        df_jb2 = pd.read_csv(p_jb2, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
        for _, r in df_jb2[df_jb2["jailbreak"].astype(str).str.strip().str.upper().isin(["TRUE", "1"])].iterrows():
            add_item(r.iloc[0], 1, "jailbreak_llms_1225")
    print(f"jailbreak_llms: gathered {len(jb_records):,} JB samples.")

    # 7. Datasets_main for label 2 (jailbreak)
    for p_dm in [
        os.path.join(dataset_dir, "Datasets_main", "dataset.csv"),
        r"C:\Users\FPT\Desktop\TF_IDF\Datasets_main\dataset.csv",
    ]:
        if os.path.exists(p_dm):
            df_dm = pd.read_csv(p_dm, encoding="utf-8", encoding_errors="replace", low_memory=False)
            c0 = len(jb_records)
            for _, r in df_dm[df_dm["label"].astype(str).str.strip().isin(["2", "2.0"])].iterrows():
                add_item(r["prompt"], 1, "datasets_main_jb")
            print(f"datasets_main ({os.path.basename(p_dm)}): added {len(jb_records) - c0:,} JB samples.")

    # 8. JailbreakDB sampling across all sources
    p_jb_db = os.path.join(dataset_dir, "JailbreakDB", "text_jailbreak_unique.csv")
    if os.path.exists(p_jb_db):
        print("Sampling diverse jailbreaks from JailbreakDB across 39 sources...")
        jb_sources_collected = []
        chunk_size = 50000
        for chunk in pd.read_csv(p_jb_db, chunksize=chunk_size, usecols=["user_prompt", "jailbreak", "source"], low_memory=False):
            sampled = chunk.groupby("source", group_keys=False).sample(frac=0.12, random_state=random_seed)
            jb_sources_collected.append(sampled)
            if sum(len(x) for x in jb_sources_collected) >= 45000:
                break
        df_jb_db = pd.concat(jb_sources_collected, ignore_index=True)
        c0 = len(jb_records)
        for _, r in df_jb_db.iterrows():
            add_item(r["user_prompt"], 1, "jailbreakdb")
        print(f"JailbreakDB: added {len(jb_records) - c0:,} JB samples.")

    print(f"Total raw JB candidates: {len(jb_records):,}")

    # Deduplicate PI
    df_pi = pd.DataFrame(pi_records).drop_duplicates(subset=["norm_key"], keep="first")
    print(f"\nUnique PI (Label 0) samples: {len(df_pi):,}")

    # Deduplicate JB
    df_jb = pd.DataFrame(jb_records).drop_duplicates(subset=["norm_key"], keep="first")
    print(f"Unique JB (Label 1) samples: {len(df_jb):,}")

    # Remove any overlap between PI and JB (prioritize PI or drop)
    overlap = set(df_pi["norm_key"]).intersection(set(df_jb["norm_key"]))
    if overlap:
        print(f"Resolving {len(overlap)} overlapping prompts between PI and JB...")
        df_jb = df_jb[~df_jb["norm_key"].isin(overlap)]

    # Balance classes so PI and JB have an optimal, equal representation
    n_samples = min(len(df_pi), len(df_jb))
    print(f"Balancing dataset to {n_samples:,} samples per class (50/50 balance)...")
    
    df_pi_balanced = df_pi.sample(n=n_samples, random_state=random_seed)
    df_jb_balanced = df_jb.sample(n=n_samples, random_state=random_seed)

    # Combine and shuffle
    df_final = pd.concat([df_pi_balanced, df_jb_balanced], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Strictly keep only 'prompt' and 'label'
    df_export = df_final[["prompt", "label"]].copy()

    # Save to primary output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_export.to_csv(output_path, index=False, encoding="utf-8")
    
    # Save copy to secondary output for convenience
    os.makedirs(os.path.dirname(secondary_output), exist_ok=True)
    df_export.to_csv(secondary_output, index=False, encoding="utf-8")

    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\n" + "=" * 65)
    print("      BINARY DATASET CREATION COMPLETE (PI vs JB)")
    print("=" * 65)
    print(f"Primary Output:   {output_path} ({sz_mb:.2f} MB)")
    print(f"Secondary Output: {secondary_output}")
    print(f"Total Rows:       {len(df_export):,}")
    print(f"Columns:          {df_export.columns.tolist()}")
    print("-" * 65)
    print("Class Distribution:")
    counts = df_export["label"].value_counts().sort_index()
    label_names = {0: "0: Prompt Injection", 1: "1: Jailbreak"}
    for lbl, cnt in counts.items():
        pct = (cnt / len(df_export)) * 100
        print(f"  - {label_names[lbl]:<22}: {cnt:>6,} ({pct:.1f}%)")
    print("-" * 65)
    print("Sample Rows (First 4):")
    for idx, row in df_export.head(4).iterrows():
        lbl_str = label_names[row['label']]
        txt = repr(str(row['prompt'])[:75])
        print(f"  [{idx + 1}] Label {lbl_str} | Text: {txt}...")
    print("=" * 65 + "\n")

    return output_path

if __name__ == "__main__":
    build_pi_vs_jb_dataset()
