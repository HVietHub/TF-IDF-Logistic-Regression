"""
Build Comprehensive Multi-Source Dataset
========================================
Consolidates and balances all datasets in C:\\Users\\FPT\\Desktop\\TF_IDF\\Dataset
into a single, high-quality, deduplicated dataset for training:
  - Class 0: benign
  - Class 1: promptinjection
  - Class 2: jailbreak

Ensures strict zero-leakage against test.csv.
"""

import os
import sys
import unicodedata
import re
import pandas as pd
import numpy as np

def normalize_text_key(text: str) -> str:
    """Normalize text for strict deduplication matching."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = unicodedata.normalize("NFC", text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def load_test_prompts(test_csv_path: str) -> set:
    if not os.path.exists(test_csv_path):
        print(f"Warning: {test_csv_path} not found. Skipping test prompt exclusion.")
        return set()
    df_test = pd.read_csv(test_csv_path)
    col = "text" if "text" in df_test.columns else df_test.columns[0]
    test_keys = set(normalize_text_key(p) for p in df_test[col].dropna())
    print(f"Loaded {len(test_keys)} unique test prompt keys to exclude from training.")
    return test_keys

def build_dataset(
    dataset_dir: str = r"C:\Users\FPT\Desktop\TF_IDF\Dataset",
    test_csv_path: str = r"C:\Users\FPT\Desktop\TF_IDF\test.csv",
    output_path: str = r"C:\Users\FPT\Desktop\TF_IDF\Dataset\Datasets_main\comprehensive_dataset.csv",
    max_jb_samples: int = 35000,
    max_reg_samples: int = 35000,
    random_seed: int = 42,
):
    np.random.seed(random_seed)
    test_keys = load_test_prompts(test_csv_path)
    
    collected_records = [] # (prompt, label, source)

    def add_records(df: pd.DataFrame, prompt_col: str, label_col_or_val, source_name: str, filter_fn=None):
        nonlocal collected_records
        count = 0
        for _, row in df.iterrows():
            prompt = str(row[prompt_col]) if pd.notnull(row[prompt_col]) else ""
            if not prompt.strip() or prompt.strip().lower() in ("nan", "none"):
                continue
            
            if filter_fn and not filter_fn(row):
                continue

            if callable(label_col_or_val):
                lbl = label_col_or_val(row)
            elif isinstance(label_col_or_val, str):
                lbl = row[label_col_or_val]
            else:
                lbl = label_col_or_val

            if lbl is None or lbl not in (0, 1, 2, "0", "1", "2"):
                continue

            lbl = int(lbl)
            norm_key = normalize_text_key(prompt)
            if norm_key in test_keys:
                continue

            collected_records.append({"prompt": prompt, "label": lbl, "source": source_name, "norm_key": norm_key})
            count += 1
        print(f"[{source_name}] Loaded {count:,} candidate samples.")

    # 1. Enterprise prompt injection dataset
    p_ent = os.path.join(dataset_dir, "enterprise-prompt-injection-dataset", "enterprise_prompt_injection.csv")
    if os.path.exists(p_ent):
        df_ent = pd.read_csv(p_ent)
        def ent_label(row):
            l = str(row["label"]).upper().strip()
            if l == "INJECTION": return 1
            if l == "SAFE": return 0
            return None
        add_records(df_ent, "text", ent_label, "enterprise_dataset")

    # 2. SPML prompt injection
    p_spml = os.path.join(dataset_dir, "Prompt_Injection", "spml_prompt_injection.csv")
    if os.path.exists(p_spml):
        df_spml = pd.read_csv(p_spml)
        def spml_label(row):
            val = str(row["Prompt injection"]).strip()
            if val in ("1", "1.0"): return 1
            if val in ("0", "0.0"): return 0
            return None
        add_records(df_spml, "User Prompt", spml_label, "spml_prompt_injection")

    # 3. prompt-injections dataset
    p_pi = os.path.join(dataset_dir, "prompt-injections", "prompt_injections.csv")
    if os.path.exists(p_pi):
        df_pi = pd.read_csv(p_pi)
        def pi_label(row):
            val = str(row["label"]).strip()
            if val in ("1", "1.0"): return 1
            if val in ("0", "0.0"): return 0
            return None
        add_records(df_pi, "text", pi_label, "prompt_injections")

    # 4. jailbreak_llms prompts
    jb_base = os.path.join(dataset_dir, "jailbreak_llms", "prompts")
    p_jb1 = os.path.join(jb_base, "jailbreak_prompts_2023_05_07.csv")
    p_jb2 = os.path.join(jb_base, "jailbreak_prompts_2023_12_25.csv")
    p_reg1 = os.path.join(jb_base, "regular_prompts_2023_05_07.csv")
    p_reg2 = os.path.join(jb_base, "regular_prompts_2023_12_25.csv")

    if os.path.exists(p_jb1):
        df_jb1 = pd.read_csv(p_jb1, on_bad_lines="skip")
        add_records(df_jb1, df_jb1.columns[0], 2, "jailbreak_llms_jb1")
    if os.path.exists(p_jb2):
        df_jb2 = pd.read_csv(p_jb2, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
        def jb2_label(row):
            v = str(row.get("jailbreak", "")).strip().upper()
            return 2 if v in ("TRUE", "1") else None
        add_records(df_jb2, df_jb2.columns[0], jb2_label, "jailbreak_llms_jb2")

    if os.path.exists(p_reg1):
        df_reg1 = pd.read_csv(p_reg1, on_bad_lines="skip")
        add_records(df_reg1, df_reg1.columns[0], 0, "jailbreak_llms_reg1")
    if os.path.exists(p_reg2):
        df_reg2 = pd.read_csv(p_reg2, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
        def reg2_label(row):
            v = str(row.get("jailbreak", "")).strip().upper()
            return 0 if v in ("FALSE", "0") else None
        add_records(df_reg2, df_reg2.columns[0], reg2_label, "jailbreak_llms_reg2")

    # 5. Datasets_main (root and subfolder)
    for p_dm in [
        os.path.join(dataset_dir, "Datasets_main", "dataset.csv"),
        r"C:\Users\FPT\Desktop\TF_IDF\Datasets_main\dataset.csv",
    ]:
        if os.path.exists(p_dm):
            df_dm = pd.read_csv(p_dm, encoding="utf-8", encoding_errors="replace", low_memory=False)
            def dm_label(row):
                try:
                    l = int(row["label"])
                    return l if l in (0, 1, 2) else None
                except:
                    return None
            add_records(df_dm, "prompt", dm_label, "datasets_main")

    # 6. JailbreakDB sampling
    p_jb_db = os.path.join(dataset_dir, "JailbreakDB", "text_jailbreak_unique.csv")
    if os.path.exists(p_jb_db):
        print("\nSampling diverse jailbreaks from JailbreakDB (6.5M pool)...")
        # Read in chunks and sample proportionally by source
        jb_sources_collected = []
        chunk_size = 50000
        for chunk in pd.read_csv(p_jb_db, chunksize=chunk_size, usecols=["user_prompt", "jailbreak", "source"], low_memory=False):
            sampled = chunk.groupby("source", group_keys=False).sample(frac=0.10, random_state=random_seed)
            jb_sources_collected.append(sampled)
            if sum(len(x) for x in jb_sources_collected) >= max_jb_samples * 1.5:
                break
        df_jb_db = pd.concat(jb_sources_collected, ignore_index=True)
        if len(df_jb_db) > max_jb_samples:
            df_jb_db = df_jb_db.sample(n=max_jb_samples, random_state=random_seed)
        add_records(df_jb_db, "user_prompt", 2, "JailbreakDB_jailbreak")

    p_reg_db = os.path.join(dataset_dir, "JailbreakDB", "text_regular_unique.csv")
    if os.path.exists(p_reg_db):
        print("\nSampling diverse benign prompts from JailbreakDB regular (5.6M pool)...")
        reg_sources_collected = []
        chunk_size = 50000
        for chunk in pd.read_csv(p_reg_db, chunksize=chunk_size, usecols=["user_prompt", "jailbreak", "source"], low_memory=False):
            sampled = chunk.groupby("source", group_keys=False).sample(frac=0.05, random_state=random_seed)
            reg_sources_collected.append(sampled)
            if sum(len(x) for x in reg_sources_collected) >= max_reg_samples * 1.5:
                break
        df_reg_db = pd.concat(reg_sources_collected, ignore_index=True)
        if len(df_reg_db) > max_reg_samples:
            df_reg_db = df_reg_db.sample(n=max_reg_samples, random_state=random_seed)
        add_records(df_reg_db, "user_prompt", 0, "JailbreakDB_regular")

    print(f"\nTotal raw collected records: {len(collected_records):,}")
    df_all = pd.DataFrame(collected_records)

    # Deduplicate by norm_key: priority to attack (1 or 2) over benign (0) if conflicting
    print("Deduplicating and resolving conflicts (attack labels take priority over benign)...")
    df_all = df_all.sort_values(by=["norm_key", "label"], ascending=[True, False])
    df_all = df_all.drop_duplicates(subset=["norm_key"], keep="first")
    print(f"After exact key deduplication: {len(df_all):,} rows.")

    # Class distribution before balancing
    print("\nClass distribution before balancing:")
    print(df_all["label"].value_counts().sort_index().to_dict())

    # Final balance:
    # Class 1 is ~13.8k samples.
    # We balance Class 0 and Class 2 reasonably (e.g. 25-30k benign, ~13.8k PI, ~25-30k JB)
    # to avoid extreme class imbalance while maximizing diversity.
    counts = df_all["label"].value_counts()
    n_pi = counts.get(1, 0)
    print(f"Prompt injection count (Class 1): {n_pi:,}")
    
    cap_other = min(35000, max(25000, int(n_pi * 2.2)))
    balanced_dfs = []
    for lbl in [0, 1, 2]:
        sub_df = df_all[df_all["label"] == lbl]
        if len(sub_df) > cap_other:
            sub_df = sub_df.sample(n=cap_other, random_state=random_seed)
        balanced_dfs.append(sub_df)
    
    df_final = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_export = df_final[["prompt", "label"]].copy()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_export.to_csv(output_path, index=False, encoding="utf-8")
    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"DATASET BUILD COMPLETE!")
    print(f"Output File: {output_path} ({sz_mb:.2f} MB)")
    print(f"Total rows: {len(df_export):,}")
    print(f"Distribution: {df_export['label'].value_counts().sort_index().to_dict()}")
    print(f"Source breakdown:")
    print(df_final["source"].value_counts().to_string())
    print(f"=======================================================\n")
    return output_path

if __name__ == "__main__":
    build_dataset()
