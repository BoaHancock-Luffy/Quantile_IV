#将连续结局数据改成二分类,并计算OR值
import os
import torch
import pandas as pd
from ml_mr.estimation.quantile_iv import fit_quantile_iv, QuantileIVEstimator
from ml_mr.utils.data import IVDatasetWithGenotypes
from pytorch_lightning import Trainer

def main():
    trainer = Trainer(num_sanity_val_steps=0)

    # -------------------------
    # 读取数据
    # -------------------------
    parquet_path = os.path.join("examples", "mr_simulation_1_sim_data.parquet")
    csv_path = os.path.join("examples", "mr_simulation_1_sim_data.csv.gz")

    if not os.path.exists(parquet_path):
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="fastparquet")

    df = pd.read_parquet(parquet_path, engine="fastparquet")
    exposure_col = "exposure"
    outcome_col = "outcome"
    iv_cols = [c for c in df.columns if c.startswith("v")]

    # 二分类 Outcome
    threshold = 0
    df[outcome_col + "_bin"] = (df[outcome_col] > threshold).astype(int)

    # 构建 IV 数据集
    dataset = IVDatasetWithGenotypes.from_dataframe(
        df,
        exposure_col=exposure_col,
        outcome_col=outcome_col + "_bin",
        iv_cols=iv_cols,
    )

    # 训练 Quantile IV
    estimator = fit_quantile_iv(
        dataset=dataset,
        n_quantiles=5,
        exposure_hidden=[128, 64],
        outcome_hidden=[64, 32],
        output_dir="results_qiv_sim1",
        binary_outcome=True
    )

    # 生成 X 范围并预测概率
    min_x, max_x = df[exposure_col].min(), df[exposure_col].max()
    xs = torch.linspace(min_x, max_x, 100).reshape(-1, 1)
    logits = estimator.iv_reg_function(xs)
    probs = torch.sigmoid(logits)

    # 计算 OR
    eps = 1e-8
    logits_eps = torch.log(probs / (1 - probs + eps))
    delta = logits_eps[1:] - logits_eps[:-1]
    or_values = torch.exp(delta)

    # 保存结果
    or_df = pd.DataFrame({
        "X_start": xs[:-1].flatten().numpy(),
        "X_end": xs[1:].flatten().numpy(),
        "OR": or_values.flatten().numpy()
    })
    or_df.to_csv("OR_results_sim1.csv", index=False)
    print("OR表格已生成：OR_results_sim1.csv")


if __name__ == "__main__":
    main()
