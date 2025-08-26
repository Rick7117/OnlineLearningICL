import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from glob import glob

def calculate_accuracy(df):
    """计算准确率"""
    if 'correct' in df.columns:
        return df['correct'].mean()
    else:
        # 如果没有correct列，根据true_label和predicted_label计算
        correct = (df['true_label'] == df['predicted_label']).sum()
        total = len(df)
        return correct / total if total > 0 else 0

def extract_k_value(filename):
    """从文件名中提取k值"""
    match = re.search(r'k(\d+)', filename)
    return int(match.group(1)) if match else None

def extract_boolean_flag(filename):
    """从文件名中提取True/False标志"""
    if 'True' in filename:
        return True
    elif 'False' in filename:
        return False
    return None

def main():
    # 设置结果目录路径
    results_dir = "/home/huangbinbin03/longicl/results"
    
    # 如果目录不存在，使用当前目录作为示例
    if not os.path.exists(results_dir):
        print(f"警告: 目录 {results_dir} 不存在，请确保路径正确")
        print("您可以修改 results_dir 变量为正确的路径")
        return
    
    # 查找所有CSV文件
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print(f"在 {results_dir} 中未找到CSV文件")
        return
    
    # 存储结果
    true_results = []  # 存储True文件的结果
    false_results = [] # 存储False文件的结果
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"处理文件: {filename}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查必要的列是否存在
            required_columns = ['text', 'true_label', 'predicted_label']
            if not all(col in df.columns for col in required_columns):
                print(f"警告: 文件 {filename} 缺少必要的列")
                continue
            
            # 计算准确率
            accuracy = calculate_accuracy(df)
            
            # 提取k值
            k_value = extract_k_value(filename)
            if k_value is None:
                print(f"警告: 无法从文件名 {filename} 中提取k值")
                continue
            
            # 根据文件名中的True/False标志分类
            is_true_file = extract_boolean_flag(filename)
            if is_true_file is True:
                true_results.append((k_value, accuracy))
            elif is_true_file is False:
                false_results.append((k_value, accuracy))
            else:
                print(f"警告: 无法从文件名 {filename} 中确定True/False标志")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
    
    # 排序结果（按k值）
    true_results.sort(key=lambda x: x[0])
    false_results.sort(key=lambda x: x[0])
    
    # 绘制图像
    plt.figure(figsize=(10, 6))
    
    # 绘制False文件的折线
    if false_results:
        false_k_values, false_accuracies = zip(*false_results)
        plt.plot(false_k_values, false_accuracies, 'o-', label='Without Reasoning', linewidth=2, markersize=6)
    
    # 绘制True文件的折线
    if true_results:
        true_k_values, true_accuracies = zip(*true_results)
        plt.plot(true_k_values, true_accuracies, 's-', label='Reasoning', linewidth=2, markersize=6)
    
    # 设置图像属性
    plt.xlabel('# of demos', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    # plt.title('Accuracy with ', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围为0-1
    plt.ylim(0, 1)
    
    # 添加数值标签
    if false_results:
        for k, acc in false_results:
            plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    if true_results:
        for k, acc in true_results:
            plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # 保存图像
    plt.tight_layout()
    output_file = 'accuracy_vs_k_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图像已保存为: {output_file}")
    
    # 显示图像
    plt.show()
    
    # 打印统计信息
    print(f"\n处理完成:")
    print(f"False文件数量: {len(false_results)}")
    print(f"True文件数量: {len(true_results)}")
    
    if false_results:
        print(f"False文件准确率范围: {min(acc for _, acc in false_results):.3f} - {max(acc for _, acc in false_results):.3f}")
    
    if true_results:
        print(f"True文件准确率范围: {min(acc for _, acc in true_results):.3f} - {max(acc for _, acc in true_results):.3f}")

if __name__ == "__main__":
    main()