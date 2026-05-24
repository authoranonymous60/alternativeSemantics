import pandas as pd
import sys
import argparse

def compute_stats(df):
    stats = {}

    test = df[df["is_few_shot"] == 0]
    few  = df[df["is_few_shot"] == 1]

    def block(prefix, subset):
        block_stats = {}
        if len(subset) == 0:
            return block_stats

        has_inf   = subset["inf_correct"].notna().any()
        has_trans = subset["trans_correct"].notna().any()

        if has_inf:
            block_stats[f"{prefix}_inference_accuracy"] = subset["inf_correct"].mean()
            k = int(subset["inf_correct"].sum())
            n = int(len(subset))
            block_stats[f"{prefix}_inference_counts"] = f"{k}/{n}"

            for foc in [1, 2]:
                s = subset[subset["focus"] == foc]
                block_stats[f"{prefix}_inference_accuracy_focus_{foc}"] = (
                    s["inf_correct"].mean() if len(s) else None
                )
            for gold in ["A", "B", "C"]:
                s = subset[subset["true_A"] == gold]
                block_stats[f"{prefix}_inference_accuracy_gold_{gold}"] = (
                    s["inf_correct"].mean() if len(s) else None
                )
            for pred in ["A", "B", "C"]:
                s = subset[subset["model_A"] == pred]
                block_stats[f"{prefix}_inference_accuracy_model_{pred}"] = (
                    s["inf_correct"].mean() if len(s) else None
                )

        if has_trans:
            block_stats[f"{prefix}_transcription_accuracy"] = subset["trans_correct"].mean()
            for foc in [1, 2]:
                s = subset[subset["focus"] == foc]
                block_stats[f"{prefix}_transcription_accuracy_focus_{foc}"] = (
                    s["trans_correct"].mean() if len(s) else None
                )

        if has_inf and has_trans:
            for tr in [0, 1]:
                s = subset[subset["trans_correct"] == tr]
                block_stats[f"{prefix}_inference_accuracy_given_transcription_{tr}"] = (
                    s["inf_correct"].mean() if len(s) else None
                )

        return block_stats

    stats.update(block("test", test))
    stats.update(block("fewshot", few))
    return stats


def compute_cross_run_stats(inf_df, trans_df):
    """Join inference and transcription CSVs on (file_id, example_index) and compute
    conditional inference accuracy given transcription correctness."""
    key = ["file_id", "example_index"]
    merged = inf_df[key + ["inf_correct", "focus", "true_A", "is_few_shot"]].merge(
        trans_df[key + ["trans_correct"]],
        on=key,
        how="inner",
    )
    test = merged[merged["is_few_shot"] == 0]
    if len(test) == 0:
        return {}

    stats = {}
    stats["cross_n_items"] = len(test)
    for tr in [0, 1]:
        s = test[test["trans_correct"] == tr]
        stats[f"test_inference_accuracy_given_transcription_{tr}"] = (
            s["inf_correct"].mean() if len(s) else None
        )
        stats[f"test_n_given_transcription_{tr}"] = len(s)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_csvs", nargs="+",
                        help="Master inference (or 'both') CSV file(s)")
    parser.add_argument("--trans", metavar="TRANS_CSV", action="append", default=[],
                        help="Master transcription CSV for cross-run inf|trans analysis (repeat for multiple)")
    args = parser.parse_args()

    dfs = []
    for f in args.inference_csvs:
        print(f"Loading {f} ...")
        df = pd.read_csv(f)
        # Coerce scored columns to numeric, treating empty strings as NaN
        for col in ["inf_correct", "trans_correct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)

    trans_df = None
    if args.trans:
        trans_dfs = []
        for f in args.trans:
            print(f"Loading transcription CSV: {f} ...")
            df = pd.read_csv(f)
            if "trans_correct" in df.columns:
                df["trans_correct"] = pd.to_numeric(df["trans_correct"], errors="coerce")
            trans_dfs.append(df)
        trans_df = pd.concat(trans_dfs, ignore_index=True)

    models = sorted(big["model_name"].unique())

    print("\n=== MODEL STATISTICS ===\n")

    for m in models:
        print(f"### Model: {m}")
        dfm = big[big["model_name"] == m]
        stats = compute_stats(dfm)
        for key, value in stats.items():
            print(f"{key}: {value}")

        if trans_df is not None:
            trans_m = trans_df[trans_df["model_name"] == m] if "model_name" in trans_df.columns else trans_df
            cross = compute_cross_run_stats(dfm, trans_m)
            if cross:
                print("--- cross-run (inf | transcription) ---")
                for key, value in cross.items():
                    print(f"{key}: {value}")
        print()


if __name__ == "__main__":
    main()
