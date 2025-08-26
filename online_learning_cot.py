import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional, Union
import time
from tqdm import tqdm
from llm_batch_processor import BaseModelResponseGenerator

class CoTReasoningGenerator(BaseModelResponseGenerator):
    """CoT推理生成器，实现在线学习过程"""
    
    def __init__(self, model_path: str, label_mapping: Dict[int, str] = None, max_examples: int = 3):
        """
        初始化CoT推理生成器
        
        Args:
            model_path: 模型路径
            label_mapping: 标签映射字典
            max_examples: 最大例子数量，防止prompt过长
        """
        super().__init__(model_path)
        self.label_mapping = label_mapping or {0: "negative", 1: "neutral", 2: "positive"}
        self.consistent_examples = []  # 存储一致的样本
        self.accuracy_history = []  # 存储准确率历史
        self.current_accuracy = 0.0
        self.total_processed = 0
        self.correct_predictions = 0
        self.max_examples = max_examples  # 最大例子数量
    
    def create_prompt(self, text: str, examples: List[Dict] = None, correction_info: Dict = None, **kwargs) -> Dict[str, str]:
        """
        创建CoT推理prompt
        
        Args:
            text: 输入文本
            examples: 过往一致样本列表
            correction_info: 纠错信息，包含previous_prediction和true_label
            **kwargs: 其他参数
            
        Returns:
            包含system_prompt和user_prompt的字典
        """
        # 构建标签选项
        label_options = []
        for label_id, label_name in sorted(self.label_mapping.items()):
            label_options.append(f"{label_id} ({label_name})")
        label_choices = ", ".join(label_options)
        
        system_prompt = "You are an expert sentiment analyst. Analyze the sentiment step by step and provide reasoning."
        
        # 构建用户prompt
        user_prompt = ""
        
        # 如果有过往一致样本，添加为示例
        if examples and len(examples) > 0:
            user_prompt += "Here are some examples of correct sentiment analysis:\n\n"
            for i, example in enumerate(examples[-3:], 1):  # 只使用最近的3个示例
                user_prompt += f"Example {i}:\n"
                user_prompt += f"Text: \"{example['text']}\"\n"
                user_prompt += f"Reasoning: {example['reasoning']}\n"
                user_prompt += f"Sentiment: {example['label']} ({self.label_mapping[example['label']]})\n\n"
        
        # 如果有纠错信息，添加纠错提示
        if correction_info:
            user_prompt += f"""IMPORTANT: You previously predicted sentiment {correction_info['previous_prediction']} ({self.label_mapping[correction_info['previous_prediction']]}) for this text with the following reasoning:
"{correction_info['previous_reasoning']}"

However, the correct sentiment is {correction_info['true_label']} ({self.label_mapping[correction_info['true_label']]}). Please re-analyze and provide better reasoning for why the sentiment should be {correction_info['true_label']} ({self.label_mapping[correction_info['true_label']]}).

"""
        
        user_prompt += f"""Now analyze the sentiment of the following text:

Text: "{text}"

Please provide your analysis in the following JSON format:
{{
    "reasoning": "Your step-by-step analysis explaining the sentiment",
    "sentiment": "Choose one number: {label_choices}"
}}"""
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析模型响应
        
        Args:
            response: 模型生成的响应
            
        Returns:
            解析后的结果字典
        """
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            
            # 验证和清理结果
            reasoning = result.get("reasoning", "").strip()
            sentiment_str = str(result.get("sentiment", "")).strip()
            
            # 提取数字标签
            predicted_label = None
            for label_id in self.label_mapping.keys():
                if str(label_id) in sentiment_str:
                    predicted_label = label_id
                    break
            
            return {
                "reasoning": reasoning,
                "predicted_label": predicted_label,
                "raw_response": response,
                "success": predicted_label is not None
            }
            
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试正则表达式提取
            import re
            
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
            sentiment_match = re.search(r'"sentiment"\s*:\s*"?([^"\n,}]+)"?', response)
            
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            sentiment_str = sentiment_match.group(1) if sentiment_match else ""
            
            predicted_label = None
            for label_id in self.label_mapping.keys():
                if str(label_id) in sentiment_str:
                    predicted_label = label_id
                    break
            
            return {
                "reasoning": reasoning,
                "predicted_label": predicted_label,
                "raw_response": response,
                "success": predicted_label is not None
            }

    def generate_single_prediction(self, text: str, correction_info: Dict = None) -> Dict[str, Any]:
        """
        为单个文本生成预测
        
        Args:
            text: 输入文本
            correction_info: 纠错信息，包含previous_prediction和true_label
            
        Returns:
            预测结果字典
        """
        # 使用过往一致样本作为示例，限制数量防止prompt过长
        if len(self.consistent_examples) > 0:
            # 选取最近的k个例子，如果总数少于k则选取所有
            examples = self.consistent_examples[-self.max_examples:]
        else:
            examples = None
        
        # 创建prompt
        prompt_dict = self.create_prompt(text=text, examples=examples, correction_info=correction_info)
        
        # 构建消息
        messages = [
            {"role": "system", "content": prompt_dict["system_prompt"]},
            {"role": "user", "content": prompt_dict["user_prompt"]}
        ]
        
        # 生成响应
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = model_inputs.input_ids.shape[1]
        response = self.tokenizer.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True
        ).strip()
        print(response)
        return self.parse_response(response)
    
    def process_online_learning(self, dataset: pd.DataFrame, max_retry: int = 3) -> pd.DataFrame:
        """
        执行在线学习过程
        
        Args:
            dataset: 包含'sentence'和'label'列的数据集
            max_retry: 最大重试次数
            
        Returns:
            包含处理结果的DataFrame
        """
        results = []
        
        print(f"开始在线学习过程，处理 {len(dataset)} 个样本...")
        
        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="在线学习进度"):
            text = row['sentence']
            true_label = row['label']
            
            retry_count = 0
            prediction_result = None
            
            # 重试循环
            correction_info = None
            while retry_count <= max_retry:
                # 生成预测
                prediction_result = self.generate_single_prediction(text, correction_info=correction_info)
                
                if not prediction_result['success']:
                    retry_count += 1
                    print(f"样本 {idx}: 解析失败，重试 {retry_count}/{max_retry}")
                    continue
                
                predicted_label = prediction_result['predicted_label']
                
                # 检查预测是否正确
                if predicted_label == true_label:
                    # 预测正确，添加到一致样本中
                    self.consistent_examples.append({
                        'text': text,
                        'label': true_label,
                        'reasoning': prediction_result['reasoning']
                    })
                    
                    # 更新统计信息
                    self.correct_predictions += 1
                    break
                else:
                    retry_count += 1
                    if retry_count <= max_retry:
                        print(f"样本 {idx}: 预测错误 (预测:{predicted_label}, 真实:{true_label})，重试 {retry_count}/{max_retry}")
                        # 设置纠错信息，下次重试时告知模型正确答案和之前的推理
                        correction_info = {
                            'previous_prediction': predicted_label,
                            'previous_reasoning': prediction_result['reasoning'],
                            'true_label': true_label
                        }
            
            # 更新总处理数量
            self.total_processed += 1
            
            # 计算当前准确率
            self.current_accuracy = self.correct_predictions / self.total_processed
            self.accuracy_history.append({
                'sample_index': idx,
                'accuracy': self.current_accuracy,
                'total_processed': self.total_processed,
                'correct_predictions': self.correct_predictions,
                'consistent_examples_count': len(self.consistent_examples)
            })
            
            # 记录结果
            result_record = {
                'sample_index': idx,
                'text': text,
                'true_label': true_label,
                'predicted_label': prediction_result['predicted_label'] if prediction_result else None,
                'reasoning': prediction_result['reasoning'] if prediction_result else None,
                'retry_count': retry_count,
                'is_correct': prediction_result['predicted_label'] == true_label if prediction_result else False,
                'current_accuracy': self.current_accuracy,
                'consistent_examples_count': len(self.consistent_examples),
                'raw_response': prediction_result['raw_response'] if prediction_result else None
            }
            
            results.append(result_record)
            
            # 添加小延迟避免过载
            time.sleep(0.1)
        
        return pd.DataFrame(results)
    
    def get_accuracy_history(self) -> pd.DataFrame:
        """
        获取准确率历史记录
        
        Returns:
            准确率历史DataFrame
        """
        return pd.DataFrame(self.accuracy_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_processed': self.total_processed,
            'correct_predictions': self.correct_predictions,
            'current_accuracy': self.current_accuracy,
            'consistent_examples_count': len(self.consistent_examples),
            'label_mapping': self.label_mapping
        }


# 使用示例
if __name__ == "__main__":
    # 导入数据加载器
    import data_loader
    
    # 加载数据集
    dataset = data_loader.FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,
        random_state=42
    )
    
    train_df, _ = dataset.get_datasets_with_label_names()
    label_mapping = dataset.get_label_mapping()
    
    # 设置模型路径
    MODEL_PATH = "/share/wangzifei03/Qwen3-4B/"
    
    # 创建CoT推理生成器
    cot_generator = CoTReasoningGenerator(
        model_path=MODEL_PATH,
        label_mapping=label_mapping,
        max_examples=5  # 限制使用例子个数，防止prompt过长
    )
    
    test_sample = train_df
    max_retry = 0
    print("开始在线学习实验...")
    results_df = cot_generator.process_online_learning(
        dataset=test_sample,
        max_retry=max_retry # 最大重新尝试制度
    )
    
    # 获取准确率历史
    accuracy_history = cot_generator.get_accuracy_history()
    
    # 获取统计信息
    stats = cot_generator.get_statistics()
    
    print("\n=== 实验结果 ===")
    print(f"总处理样本数: {stats['total_processed']}")
    print(f"正确预测数: {stats['correct_predictions']}")
    print(f"最终准确率: {stats['current_accuracy']:.4f}")
    print(f"一致样本数: {stats['consistent_examples_count']}")
    
    # 保存结果
    results_df.to_csv(f'cot_online_learning_results_retry{max_retry}.csv', index=False)
    # results_df.to_csv('cot_online_learning_results.csv', index=False)
    accuracy_history.to_csv('accuracy_history.csv', index=False)
    
    print("\n结果已保存到 'cot_online_learning_results.csv' 和 'accuracy_history.csv'")
    
    # 显示准确率变化趋势
    print("\n=== 准确率变化趋势 ===")
    for i in range(0, len(accuracy_history), max(1, len(accuracy_history)//10)):
        row = accuracy_history.iloc[i]
        print(f"样本 {int(row['sample_index']):2d}: 准确率 {row['accuracy']:.4f}, 一致样本数 {row['consistent_examples_count']}")