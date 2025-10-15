#!/usr/bin/env python3
"""
数据预处理脚本：清理重复文件、统一命名
"""
import os
import shutil
from pathlib import Path

# 数据集路径
DATA_SOURCE = "/Users/ding/Downloads/DSPdataset_exp1"
DATA_TARGET = "/Users/ding/Downloads/speech_data_cleaned"

def clean_dataset():
    """清理并整理数据集"""

    print("="*60)
    print("数据集清理与整理")
    print("="*60)

    # 创建目标目录
    os.makedirs(DATA_TARGET, exist_ok=True)

    # 获取所有类别文件夹
    categories = sorted([d for d in os.listdir(DATA_SOURCE)
                        if os.path.isdir(os.path.join(DATA_SOURCE, d))
                        and not d.startswith('.')])

    print(f"\n找到 {len(categories)} 个类别: {categories}\n")

    total_removed = 0
    total_kept = 0

    for category in categories:
        source_dir = os.path.join(DATA_SOURCE, category)
        target_dir = os.path.join(DATA_TARGET, category)

        # 创建目标类别文件夹
        os.makedirs(target_dir, exist_ok=True)

        # 获取所有wav文件
        wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]

        # 分离原始文件和重复文件
        original_files = [f for f in wav_files if not f.endswith('_1.wav')]
        duplicate_files = [f for f in wav_files if f.endswith('_1.wav')]

        print(f"类别 '{category}':")
        print(f"  原始文件: {len(original_files)}")
        print(f"  重复文件(_1): {len(duplicate_files)}")

        # 复制原始文件到目标目录，并统一命名
        for idx, filename in enumerate(sorted(original_files), 1):
            source_path = os.path.join(source_dir, filename)

            # 统一命名格式：类别名_编号.wav
            new_filename = f"{category}_{idx:03d}.wav"
            target_path = os.path.join(target_dir, new_filename)

            # 复制文件
            shutil.copy2(source_path, target_path)

        total_removed += len(duplicate_files)
        total_kept += len(original_files)

        print(f"  → 保留 {len(original_files)} 个文件")
        print()

    print("="*60)
    print("清理完成！")
    print(f"总共保留: {total_kept} 个文件")
    print(f"总共删除: {total_removed} 个重复文件")
    print(f"清理后的数据集路径: {DATA_TARGET}")
    print("="*60)

    return DATA_TARGET


def update_config(data_path):
    """更新config.py中的数据路径"""

    config_path = os.path.join(os.path.dirname(__file__), 'config.py')

    print(f"\n更新配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换DATA_DIR配置
    import re

    # 查找DATA_DIR的配置行
    pattern = r"DATA_DIR = os\.environ\.get\('SPEECH_DATA_DIR', os\.path\.join\(BASE_DIR, 'data'\)\)"
    replacement = f"DATA_DIR = os.environ.get('SPEECH_DATA_DIR', r'{data_path}')"

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✓ 配置文件已更新")
    else:
        print("⚠ 警告: 未找到DATA_DIR配置行，请手动更新config.py")


def show_dataset_summary(data_path):
    """显示数据集摘要"""

    print("\n" + "="*60)
    print("数据集摘要")
    print("="*60)

    categories = sorted([d for d in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, d))
                        and not d.startswith('.')])

    for category in categories:
        category_dir = os.path.join(data_path, category)
        wav_files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
        print(f"  {category}: {len(wav_files)} 个样本")

    print("="*60)


if __name__ == '__main__':
    # 清理数据集
    cleaned_data_path = clean_dataset()

    # 更新配置
    update_config(cleaned_data_path)

    # 显示摘要
    show_dataset_summary(cleaned_data_path)

    print("\n✓ 数据准备完成！现在可以运行实验了：")
    print("  python run.py\n")
