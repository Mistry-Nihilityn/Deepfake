import os
import json
import pandas as pd
import re


def analyze_experiment_results(root_dir, output_csv='experiment_results.csv'):
    """
    统计实验结果并生成CSV文件

    参数:
    root_dir: 实验根目录路径
    output_csv: 输出的CSV文件名
    """

    all_results = []

    target_metrics = ['ACC', 'AUC', 'Recall', 'Precision']

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            epoch_match = re.search(r'epoch[\s_]?(\d+)', item, re.IGNORECASE)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))

                methods = ['SageMaker', 'starganv2']
                for method in methods:
                    method_path = os.path.join(item_path, method)
                    json_path = os.path.join(method_path, 'test_metrics.json')

                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                metrics = json.load(f)

                            result = {
                                'epoch': epoch_num,
                                'method': method,
                            }

                            for metric in target_metrics:
                                if metric in metrics:
                                    result[metric] = metrics[metric]
                                else:
                                    # 如果指标不存在，用NaN填充
                                    result[metric] = None


                            all_results.append(result)

                        except (json.JSONDecodeError, IOError) as e:
                            print(f"警告: 无法读取 {json_path}: {e}")
                    else:
                        print(f"警告: {json_path} 不存在")

    if not all_results:
        print("错误: 未找到任何实验结果")
        return None

    df = pd.DataFrame(all_results)

    df = df.sort_values(['epoch', 'method']).reset_index(drop=True)

    df.to_csv(output_csv, index=False)

    print(f"成功处理 {len(df)} 条记录")
    print(f"epoch范围: {df['epoch'].min()} ~ {df['epoch'].max()}")
    print(f"包含的方法: {df['method'].unique().tolist()}")
    print(f"结果已保存到: {output_csv}")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print("\n数值列统计摘要:")
    print(df[numeric_cols].describe().round(4))

    return df


def generate_pivot_tables(df, output_prefix='experiment_summary'):
    """
    生成透视表，便于分析

    参数:
    df: 原始数据DataFrame
    output_prefix: 输出文件前缀
    """

    metrics = ['ACC', 'AUC', 'Recall', 'Precision']

    for metric in metrics:
        if metric in df.columns:
            pivot_df = df.pivot_table(
                values=metric,
                index='epoch',
                columns='method',
                aggfunc='mean'
            )

            pivot_file = f"{output_prefix}_{metric}.csv"
            pivot_df.to_csv(pivot_file)
            print(f"透视表已保存到: {pivot_file}")

            print(f"\n{metric} 统计:")
            print(pivot_df.describe().round(4))

    if {'epoch', 'method'}.issubset(df.columns):
        all_metrics = [col for col in df.columns if col not in ['epoch', 'method']]

        summary_list = []
        for (epoch, method), group in df.groupby(['epoch', 'method']):
            summary = {'epoch': epoch, 'method': method}
            for metric in all_metrics:
                if metric in group.columns:
                    summary[metric] = group[metric].mean()
            summary_list.append(summary)

        summary_df = pd.DataFrame(summary_list)
        summary_file = f"{output_prefix}_all_metrics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n完整汇总表已保存到: {summary_file}")


# 使用示例
if __name__ == "__main__":
    experiment_root = "log/resnet/All42/2026-01-14-21-16-14"

    results_df = analyze_experiment_results(
        root_dir=experiment_root,
        output_csv='experiment_metrics.csv'
    )

    if results_df is not None:
        generate_pivot_tables(results_df, 'experiment_pivot')

        import matplotlib.pyplot as plt

        methods = results_df['method'].unique()
        metrics = ['ACC', 'AUC', 'Recall', 'Precision']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for method in methods:
                method_data = results_df[results_df['method'] == method]
                ax.plot(method_data['epoch'], method_data[metric],
                        marker='o', label=method, linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('experiment_trends.png', dpi=150, bbox_inches='tight')
        print("趋势图已保存为: experiment_trends.png")
        plt.show()