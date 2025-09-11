import pandas as pd

def select_best_epoch(csv_path, metrics, lower_is_better, use_multiplicative=True):
    """
    Select the best epoch based on aggregated metric rankings.

    Args:
        csv_path (str): Path to CSV file containing metrics.
        metrics (list): List of metric column names in the CSV.
        lower_is_better (list of bool): True if corresponding metric is lower-is-better.
        use_multiplicative (bool): Whether to multiply ranks (True) or sum ranks (False).

    Returns:
        int: Best epoch number.
    """

    df = pd.read_csv(csv_path)

    # --- Rank each metric ---
    for metric, lower_is_better in zip(metrics, lower_is_better):
        df[metric + "_rank"] = df[metric].rank(ascending=lower_is_better, method="min")

    # --- Compute total rank ---
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    if use_multiplicative:
        df["total_rank"] = df[rank_cols].prod(axis=1)
    else:
        df["total_rank"] = df[rank_cols].sum(axis=1)

    # --- Sort by total rank (ascending = better) ---
    best_row = df.sort_values("total_rank", ascending=True).iloc[0]

    return int(best_row["epoch"])

if __name__ == "__main__":
    metrics = ["psnr", "ms_ssim", "ms_gmsd"]
    lower_flags = [False, False, True]
    best_epoch = select_best_epoch("model_select/val_metrics_bc_64_sep10.csv", metrics, lower_flags)
    print(best_epoch)
