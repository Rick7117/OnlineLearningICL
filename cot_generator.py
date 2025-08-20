import data_loader
from utils import SentimentAndReasoningGenerator
import pandas as pd
from tqdm import tqdm
import time

def process_dataframe_with_cot(
    train_df: pd.DataFrame, 
    model_path: str, 
    label_prefix: str, 
    text_prefix: str,
    label_mapping: dict,
    batch_size: int = 1,
    if_check: bool = True,
    max_retries: int = 3
    ) -> pd.DataFrame:
    """
    处理整个DataFrame，添加reason列
    
    Args:
        train_df: 包含sentence, label, label_name列的DataFrame
        model_path: 模型路径
        batch_size: 批处理大小（建议为1以避免内存问题）
        
    Returns:
        添加了reason列的DataFrame
    """
    # 创建分析器
    analyzer = SentimentAndReasoningGenerator(model_path)
    
    # 复制DataFrame
    result_df = train_df.copy()
    result_df['reason'] = None
    result_df['predicted_label'] = None
    result_df['raw_response'] = None
    
    print(f"开始处理 {len(train_df)} 条数据...")
    
    # 逐行处理
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="分析情感"):
        # try:
        # 分析情感
        result = analyzer.analyze_sentiment(row['sentence'], label_prefix, text_prefix, label_mapping)
        
        # 记录预测结果
        result_df.at[idx, 'predicted_label'] = result['predicted_label']
        result_df.at[idx, 'raw_response'] = result['raw_response']
        
        if not if_check:
            # 如果不需要检查，直接保存reasoning
            result_df.at[idx, 'reason'] = result['reasoning']
        else:
            # 需要检查预测标签与真实标签是否一致
            retry_count = 0
            
            while retry_count < max_retries:
                print(f"result type: {type(result)}")
                print(f"result content: {result}")
                print(f"row type: {type(row)}")
                print(f"row content: {row}")                
                if result['predicted_label'] == row['label']:
                    # 预测正确，保存reasoning
                    result_df.at[idx, 'reason'] = result['reasoning']
                    break
                else:
                    # 预测错误，生成纠正prompt
                    retry_count += 1
                    print(f"  预测错误，第 {retry_count} 次重试...")
                    
                    correction_prompt = analyzer.create_correction_prompt(
                        sentence=row['sentence'],
                        reasoning=result['reasoning'],
                        predicted_label=result['predicted_label'],
                        true_label=row['label'],
                        label_mapping=label_mapping
                    )
                    
                    # 重新生成回复
                    response = analyzer.generate_response(correction_prompt)
                    result = analyzer.parse_response(response, label_mapping)
            
            # 如果达到最大重试次数仍未成功，记录失败信息
            if retry_count >= max_retries and result['predicted_label'] != row['label']:
                result_df.at[idx, 'reason'] = f"重试{max_retries}次后仍预测错误: {result['reasoning']}"
                print(f"  警告: 第 {idx+1} 行达到最大重试次数，预测仍不正确")
        
        # 添加小延迟避免过载
        time.sleep(0.1)
            
        # except Exception as e:
        #     print(f"处理第 {idx} 行时出错: {e}")
        #     continue
    
    return result_df

def analyze_results(result_df: pd.DataFrame) -> dict:
    """
    分析结果统计
    
    Args:
        result_df: 处理后的DataFrame
        
    Returns:
        统计信息字典
    """
    total_samples = len(result_df)
    valid_predictions = result_df['predicted_label'].notna().sum()
    correct_predictions = (result_df['predicted_label'] == result_df['label']).sum()
    with_reason = result_df['reason'].notna().sum()
    
    accuracy = correct_predictions / valid_predictions if valid_predictions > 0 else 0
    
    stats = {
        'total_samples': total_samples,
        'valid_predictions': valid_predictions,
        'correct_predictions': correct_predictions,
        'samples_with_reason': with_reason,
        'accuracy': accuracy,
        'reason_coverage': with_reason / total_samples
    }
    
    return stats

# 使用示例
if __name__ == "__main__":
    # 假设您已经有了train_df
    dataset = data_loader.FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,  # 8:2划分
        random_state=42
    )
    train_df, _ = dataset.get_datasets_with_label_names()
    print(train_df.columns)
    text_prefix, label_prefix, label_mapping = dataset.text_prefix, dataset.label_prefix, dataset.get_label_mapping()
    if_check = True 
    # 设置模型路径（请替换为您的实际路径）
    MODEL_PATH = "/share/wangzifei03/Qwen3-4B/"
    
    # 处理数据

    print(f"开始处理测试数据..., if_check={if_check}")
    result_df = process_dataframe_with_cot(train_df.copy(), MODEL_PATH, label_prefix, text_prefix, label_mapping, if_check)
    # result_df = process_dataframe_with_cot(train_df, MODEL_PATH, label_prefix, text_prefix, label_mapping, if_check)
    
    # 分析结果
    stats = analyze_results(result_df)
    print("\n处理结果统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 显示一些示例结果
    print("\n示例结果:")
    sample_results = result_df[result_df['reason'].notna()].head(3)
    for idx, row in sample_results.iterrows():
        print(f"\n句子: {row['sentence']}")
        print(f"真实标签: {row['label']} ({row['label_name']})")
        print(f"预测标签: {row['predicted_label']}")
        print(f"推理过程: {row['reason']}")
    
    # 保存结果
    result_df.to_csv(f'sentiment_analysis_results_if_check_{if_check}.csv', index=False)
    print(f"\n结果已保存到sentiment_analysis_results_if_check_{if_check}.csv")

