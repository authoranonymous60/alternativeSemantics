import pandas as pd
import sys

def compute_stats(df):
    stats = {}

    # Split into few-shot and real test examples
    test = df[df["is_few_shot"] == 0]
    few = df[df["is_few_shot"] == 1]

    # ——— (A) Test-only evaluation ———
    def block(prefix, subset):
        block_stats = {}
        if len(subset) == 0:
            return block_stats

        block_stats[f"{prefix}_inference_accuracy"] = subset["inf_correct"].mean()

        k = int(subset["inf_correct"].sum())
        n = int(len(subset))
        block_stats[f"{prefix}_inference_counts"] = f"{k}/{n}"
        
        block_stats[f"{prefix}_transcription_accuracy"] = subset["trans_correct"].mean()

        # by gold focus (1 = first, 2 = second)
        for foc in [1, 2]:
            s = subset[subset["focus"] == foc]
            block_stats[f"{prefix}_inference_accuracy_focus_{foc}"] = (
                s["inf_correct"].mean() if len(s) else None
            )
            block_stats[f"{prefix}_transcription_accuracy_focus_{foc}"] = (
                s["trans_correct"].mean() if len(s) else None
            )

        # inference accuracy given transcription correctness
        for tr in [0, 1]:
            s = subset[subset["trans_correct"] == tr]
            block_stats[f"{prefix}_inference_accuracy_given_transcription_{tr}"] = (
                s["inf_correct"].mean() if len(s) else None
            )

        # by gold label
        for gold in ["A", "B", "C"]:
            s = subset[subset["true_A"] == gold]
            block_stats[f"{prefix}_inference_accuracy_gold_{gold}"] = (
                s["inf_correct"].mean() if len(s) else None
            )

        # by model prediction
        for pred in ["A", "B", "C"]:
            s = subset[subset["model_A"] == pred]
            block_stats[f"{prefix}_inference_accuracy_model_{pred}"] = (
                s["inf_correct"].mean() if len(s) else None
            )

        return block_stats

    # Build all stats blocks
    stats.update(block("test", test))
    stats.update(block("fewshot", few))
#    stats.update(block("combined", df))

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python stats_from_master_csvs.py master1.csv master2.csv ...")
        return

    all_files = sys.argv[1:]

    # Load and merge master CSVs
    dfs = []
    for f in all_files:
        print(f"Loading {f} ...")
        df = pd.read_csv(f)
        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)

    # Group by model_name
    models = sorted(big["model_name"].unique())

    print("\n=== MODEL STATISTICS ===\n")

    for m in models:
        print(f"### Model: {m}")
        dfm = big[big["model_name"] == m]

        stats = compute_stats(dfm)

        for key, value in stats.items():
            print(f"{key}: {value}")
        print()


if __name__ == "__main__":
    main()
