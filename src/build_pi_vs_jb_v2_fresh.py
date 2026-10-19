"""
Build Fresh Binary Dataset: Prompt Injection vs Jailbreak (V2)
==============================================================
Requirements:
1. Synthesize all newly converted CSV datasets + existing datasets.
2. Only 2 classes:
     - 0: Prompt Injection
     - 1: Jailbreak
3. Only 2 columns: ['prompt', 'label']
4. STRICT CONDITION: All Jailbreak prompts must be FRESH and NOT duplicate
   with any Jailbreak prompts in the previously generated dataset (pi_vs_jb_dataset.csv).
5. Zero leakage: Strictly exclude all prompts in test.csv.
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
    print(f"Loaded {len(test_keys):,} test keys to exclude.")
    return test_keys

def load_previous_jb_prompts(prev_dataset_path: str) -> set:
    if not os.path.exists(prev_dataset_path):
        print("No previous dataset found.")
        return set()
    df_prev = pd.read_csv(prev_dataset_path)
    # in pi_vs_jb_dataset.csv, label was 1 for jailbreak
    df_jb = df_prev[df_prev["label"] == 1]
    prev_keys = set(normalize_text_key(p) for p in df_jb["prompt"].dropna())
    print(f"Loaded {len(prev_keys):,} previously used Jailbreak keys to ensure 0 duplicates.")
    return prev_keys

def build_v2_dataset(
    dataset_dir: str = "../Dataset",
    test_csv_path: str = "data/test.csv",
    prev_dataset_path: str = "data/pi_vs_jb_dataset.csv",
    output_path: str = "data/pi_vs_jb_dataset_v2.csv",
    primary_output: str = "data/pi_vs_jb_dataset.csv",
    secondary_output: str = "../Dataset/pi_vs_jb_dataset.csv",
    random_seed: int = 42,
):
    np.random.seed(random_seed)
    test_keys = load_test_prompts(test_csv_path)
    prev_jb_keys = load_previous_jb_prompts(prev_dataset_path)

    pi_records = []
    jb_records = []

    def add_pi(prompt_str, source_name):
        p = str(prompt_str).strip()
        if not p or p.lower() in ("nan", "none") or len(p) < 5:
            return
        k = normalize_text_key(p)
        if k in test_keys:
            return
        pi_records.append({"prompt": p, "label": 0, "norm_key": k, "source": source_name})

    def add_jb(prompt_str, source_name):
        p = str(prompt_str).strip()
        if not p or p.lower() in ("nan", "none") or len(p) < 5:
            return
        k = normalize_text_key(p)
        if k in test_keys or k in prev_jb_keys:
            # STRICT REQUIREMENT: exclude test prompts and exclude previously used JB prompts
            return
        jb_records.append({"prompt": p, "label": 1, "norm_key": k, "source": source_name})

    print("\n" + "="*60)
    print("STEP 1: Collecting ALL Prompt Injection Sources (Label 0)")
    print("="*60)

    # 1. safe-guard-prompt-injection
    p_sg = os.path.join(dataset_dir, "safe-guard-prompt-injection", "safe_guard_prompt_injection.csv")
    if os.path.exists(p_sg):
        df_sg = pd.read_csv(p_sg)
        c0 = len(pi_records)
        for _, r in df_sg[df_sg["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
            add_pi(r["text"], "safe_guard")
        print(f"safe-guard-prompt-injection: added {len(pi_records) - c0:,} PI samples.")

    # 2. prompt-injection-eval-suite (16 CSV files)
    p_suite = os.path.join(dataset_dir, "prompt-injection-eval-suite", "data")
    if os.path.exists(p_suite):
        c0 = len(pi_records)
        for f in sorted(os.listdir(p_suite)):
            if f.endswith(".csv"):
                fp = os.path.join(p_suite, f)
                df_cur = pd.read_csv(fp)
                # check if label 1 is injection
                if "simsonsun_jailbreaks" in f:
                    # this is jailbreak!
                    for _, r in df_cur[df_cur["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
                        add_jb(r["text"], "simsonsun_jailbreak")
                else:
                    if "label" in df_cur.columns and "text" in df_cur.columns:
                        for _, r in df_cur[df_cur["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
                            add_pi(r["text"], f"eval_suite_{f}")
        print(f"prompt-injection-eval-suite: added {len(pi_records) - c0:,} PI samples.")

    # 3. SPML prompt injection
    p_spml = os.path.join(dataset_dir, "Prompt_Injection", "spml_prompt_injection.csv")
    if os.path.exists(p_spml):
        df_spml = pd.read_csv(p_spml)
        c0 = len(pi_records)
        for _, r in df_spml[df_spml["Prompt injection"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
            add_pi(r["User Prompt"], "spml")
        print(f"SPML: added {len(pi_records) - c0:,} PI samples.")

    # 4. enterprise-prompt-injection-dataset
    p_ent = os.path.join(dataset_dir, "enterprise-prompt-injection-dataset", "enterprise_prompt_injection.csv")
    if os.path.exists(p_ent):
        df_ent = pd.read_csv(p_ent)
        c0 = len(pi_records)
        for _, r in df_ent[df_ent["label"].astype(str).str.strip().str.upper() == "INJECTION"].iterrows():
            add_pi(r["text"], "enterprise")
        print(f"Enterprise: added {len(pi_records) - c0:,} PI samples.")

    # 5. prompt-injections
    p_pi = os.path.join(dataset_dir, "prompt-injections", "prompt_injections.csv")
    if os.path.exists(p_pi):
        df_pi = pd.read_csv(p_pi)
        c0 = len(pi_records)
        for _, r in df_pi[df_pi["label"].astype(str).str.strip().isin(["1", "1.0"])].iterrows():
            add_pi(r["text"], "prompt_injections")
        print(f"prompt-injections: added {len(pi_records) - c0:,} PI samples.")

    # 6. PIGuard train.json
    p_piguard = os.environ.get("PIGUARD_TRAIN_JSON", "data/piguard_train.json")
    if os.path.exists(p_piguard):
        import json
        with open(p_piguard, "r", encoding="utf-8") as fp:
            piguard_data = json.load(fp)
        c0 = len(pi_records)
        for item in piguard_data:
            if str(item.get("label", "")).strip() in ("1", "1.0"):
                add_pi(item.get("prompt", ""), f"piguard_{item.get('source', 'unknown')}")
        print(f"PIGuard train.json: added {len(pi_records) - c0:,} PI samples.")

    # Deduplicate PI
    df_pi_all = pd.DataFrame(pi_records).drop_duplicates(subset=["norm_key"], keep="first")
    print(f"\n=> Total UNIQUE Prompt Injection samples: {len(df_pi_all):,}")

    target_per_class = len(df_pi_all)

    print("\n" + "="*60)
    print("STEP 2: Collecting FRESH, NON-DUPLICATE Jailbreaks (Label 1)")
    print(f"Target count: ~{target_per_class:,} fresh Jailbreak samples")
    print("="*60)

    # 7. Sampling FRESH Jailbreaks from JailbreakDB (6.5M lines pool)
    p_jb_db = os.path.join(dataset_dir, "JailbreakDB", "text_jailbreak_unique.csv")
    if os.path.exists(p_jb_db):
        print("Scanning and sampling fresh jailbreaks from JailbreakDB...")
        chunk_size = 50000
        # We start reading further into the file or use a different random fraction to get fresh rows
        chunks_collected = 0
        for chunk in pd.read_csv(p_jb_db, chunksize=chunk_size, usecols=["user_prompt", "jailbreak", "source"], low_memory=False):
            chunks_collected += 1
            # Sample across sources with random_state=999 (different seed & higher fraction)
            sampled = chunk.groupby("source", group_keys=False).sample(frac=0.15, random_state=999)
            for _, r in sampled.iterrows():
                add_jb(r["user_prompt"], "jailbreakdb_fresh")
            if len(set(x["norm_key"] for x in jb_records)) >= target_per_class * 1.3:
                break
            if chunks_collected >= 30: # scanned 1.5 million rows
                break

    # Deduplicate JB
    df_jb_all = pd.DataFrame(jb_records).drop_duplicates(subset=["norm_key"], keep="first")
    print(f"=> Total FRESH, UNIQUE Jailbreak samples: {len(df_jb_all):,}")

    # Check intersection with previous JB keys:
    overlap_with_prev = set(df_jb_all["norm_key"]).intersection(prev_jb_keys)
    print(f"Overlap with previously generated Jailbreak dataset: {len(overlap_with_prev)} (MUST BE 0!)")
    assert len(overlap_with_prev) == 0, "Assertion failed: overlap with previous jailbreaks detected!"

    # Resolve any overlap between PI and JB
    pi_jb_overlap = set(df_pi_all["norm_key"]).intersection(set(df_jb_all["norm_key"]))
    if pi_jb_overlap:
        print(f"Resolving {len(pi_jb_overlap)} prompts in both PI and JB...")
        df_jb_all = df_jb_all[~df_jb_all["norm_key"].isin(pi_jb_overlap)]

    # Balance 50/50
    final_n = min(len(df_pi_all), len(df_jb_all))
    print(f"\nBalancing final dataset to {final_n:,} samples per class (Total: {final_n * 2:,})...")

    df_pi_bal = df_pi_all.sample(n=final_n, random_state=random_seed)
    df_jb_bal = df_jb_all.sample(n=final_n, random_state=random_seed)

    df_final = pd.concat([df_pi_bal, df_jb_bal], ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_export = df_final[["prompt", "label"]].copy()

    # Save to all target locations
    for out_p in [output_path, primary_output, secondary_output]:
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        df_export.to_csv(out_p, index=False, encoding="utf-8")
        sz_mb = os.path.getsize(out_p) / (1024 * 1024)
        print(f"Saved: {out_p} ({sz_mb:.2f} MB)")

    print("\n" + "=" * 65)
    print("      FINAL DATASET SUMMARY (PI vs JB - FRESH VERSION)")
    print("=" * 65)
    print(f"Total Rows:  {len(df_export):,}")
    print(f"Columns:     {df_export.columns.tolist()}")
    print("Class Distribution:")
    counts = df_export["label"].value_counts().sort_index()
    names = {0: "0: Prompt Injection", 1: "1: Jailbreak (Fresh)"}
    for lbl, cnt in counts.items():
        pct = (cnt / len(df_export)) * 100
        print(f"  - {names[lbl]:<25}: {cnt:>6,} ({pct:.1f}%)")
    print("=" * 65 + "\n")

    return output_path

if __name__ == "__main__":
    build_v2_dataset()
