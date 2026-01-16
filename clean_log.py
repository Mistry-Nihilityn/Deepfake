import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def is_timestamp_folder(folder_name: str) -> bool:
    """检查文件夹名是否为时间戳格式：YYYY-MM-DD-HH-MM-SS"""
    pattern = r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'
    return bool(re.match(pattern, folder_name))


def check_strict_mode(exp_path: Path) -> bool:
    """严格模式检查：是否存在 experiment_finished.txt 且内容为 YES"""
    finished_file = exp_path / "experiment_finished.txt"
    if not finished_file.exists():
        return False

    try:
        with open(finished_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content.upper() == "YES"
    except Exception:
        return False


def check_gentle_mode(exp_path: Path) -> bool:
    """温和模式检查：存在特定结果文件夹为有效，只有中间输出文件夹为无效"""
    epoch_pattern = [re.compile(r'^epoch\d+$'), re.compile(r'^test_epoch_\d+$')]

    has_result_folder = False
    has_only_intermediate_folders = True

    for item in exp_path.iterdir():
        if item.is_dir():
            folder_name = item.name

            if folder_name.startswith('.'):
                continue

            is_intermediate = (
                    folder_name == 'train_output' or
                    folder_name == 'val_output' or
                    folder_name == 'final_test' or
                    any([bool(pattern.match(folder_name)) for pattern in epoch_pattern])
            )

            if not is_intermediate:
                print(f"Find {item}")
                has_result_folder = True
                has_only_intermediate_folders = False
                break
            else:
                pass

    if has_result_folder:
        return True
    elif has_only_intermediate_folders and any(exp_path.iterdir()):
        return False
    else:
        return False


def analyze_experiment_results(
        log_dir: str,
        mode: str = "strict",
        verbose: bool = True
) -> Dict[str, List[str]]:
    """
    分析实验日志目录，识别无效实验结果

    Args:
        log_dir: 日志目录路径
        mode: 检查模式，"strict" 或 "gentle"
        dry_run: 是否仅模拟运行（不实际删除）
        verbose: 是否打印详细信息

    Returns:
        包含删除和保留列表的字典
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"错误：目录 {log_dir} 不存在")
        return {"to_delete": [], "to_keep": []}

    # 递归查找所有子目录
    timestamp_folders = []
    for item in log_path.rglob('*'):
        if item.is_dir() and is_timestamp_folder(item.name):
            timestamp_folders.append(item)

    if verbose:
        print(f"在 {log_dir} 中找到 {len(timestamp_folders)} 个实验文件夹")

    # 检查每个实验文件夹
    to_delete = []
    to_keep = []

    for exp_folder in timestamp_folders:
        if mode == "strict":
            is_valid = check_strict_mode(exp_folder)
        else:  # gentle mode
            is_valid = check_gentle_mode(exp_folder)

        if is_valid:
            to_keep.append(str(exp_folder))
        else:
            to_delete.append(str(exp_folder))

    # 打印结果
    if verbose:
        print("\n" + "=" * 60)
        print(f"检查模式: {mode}")
        print(f"有效实验: {len(to_keep)} 个")
        print(f"无效实验: {len(to_delete)} 个")

        if to_keep:
            print("\n✅ 将保留的实验:")
            for i, path in enumerate(to_keep):
                print(f"  {i}. {path}")

        if to_delete:
            print("\n🗑️  将删除的实验:")
            for i, path in enumerate(to_delete):
                print(f"  {i}. {path}")
        print("=" * 60)

    # 计算总大小
    if to_delete:
        total_size = 0
        for folder in to_delete:
            folder_path = Path(folder)
            total_size += sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())

        if verbose:
            print(f"\n📊 预计释放空间: {total_size / 1024 ** 2:.2f} MB")

    print("\n⚠️  警告：这将在磁盘上实际删除文件！")
    confirm = input(f"确认删除 {len(to_delete)} 个实验文件夹？(输入 'yes' 确认): ")

    if confirm.lower() == 'yes':
        print("开始删除...")
        for folder in to_delete:
            try:
                shutil.rmtree(folder)
                print(f"  已删除: {os.path.basename(folder)}")
            except Exception as e:
                print(f"  删除 {folder} 时出错: {e}")
        print("删除完成！")
    else:
        print("操作已取消")

    return {
        "to_delete": to_delete,
        "to_keep": to_keep,
        "mode": mode,
        "total_folders": len(timestamp_folders)
    }


if __name__ == "__main__":
    log_path = "log"
    results = analyze_experiment_results(log_path, mode="gentle")
