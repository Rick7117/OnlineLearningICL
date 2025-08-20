import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import data_loader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from CONSTANT import SHOT_SEPARATOR, COT_PREFIX

class InContextLearningClassifier:
    def __init__(self, train_df, test_df, label_mapping, text_prefix, label_prefix, model_path: str, k: int = 3, random_state: int = 42):
        """
        初始化In-Context Learning分类器
        
        Args:
            model_path: 模型路径
            k: 从训练集中采样的示例个数
            random_state: 随机种子
        """
        self.k = k
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 初始化模型和tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.train_df, self.test_df = train_df, test_df
        self.label_mapping = label_mapping
        self.text_prefix, self.label_prefix = text_prefix, label_prefix
        
        print(f"训练集大小: {len(self.train_df)}")
        print(f"测试集大小: {len(self.test_df)}")
        print(f"标签映射: {self.label_mapping}")
        print(f"K值 (示例个数): {self.k}")
    
    def make_one_shot_prompt(self, text: str, reasoning: str, label: str) -> str:
        """
        制作单个示例的prompt
        
        Args:
            text: 输入文本
            reasoning: 推理过程（可选）
            label: 标签（可选，推理时可能为空）
            text_prefix: 文本前缀
            label_prefix: 标签前缀
            
        Returns:
            格式化的prompt字符串
        """
        # 构建基础部分：text_prefix + text + \n
        prompt = self.text_prefix + text + "\n"
        
        if reasoning is not None and reasoning != "" and not pd.isna(reasoning):
            prompt += COT_PREFIX + str(reasoning) + "\n"

        # 如果label不为空，添加label部分
        if label is not None and label != "" and not pd.isna(label):
            prompt += self.label_prefix + str(label)
        
        return prompt
    
    def make_long_shot_prompt(self, shot_prompts: list) -> str:
        """
        将多个one shot prompt拼接成长文本
        
        Args:
            shot_prompts: 包含多个one shot prompt的列表
            
        Returns:
            使用分隔符拼接的长文本prompt
        """
        if not shot_prompts:
            return ""
        
        # 过滤掉空的prompt
        valid_prompts = [prompt for prompt in shot_prompts if prompt and prompt.strip()]
        
        if not valid_prompts:
            return ""
        
        # 使用分隔符拼接所有prompt
        return SHOT_SEPARATOR.join(valid_prompts)
    
    def sample_examples(self, target_text: str, k: int = None) -> List[Dict]:
        """
        从训练集中随机采样k个示例作为in-context learning的例子
        
        Args:
            target_text: 目标文本（避免采样到相同的文本）
            k: 采样个数，如果为None则使用self.k
            
        Returns:
            采样的示例列表
        """
        if k is None:
            k = self.k
        
        # 过滤掉与目标文本相同的样本
        available_samples = self.train_df[self.train_df['sentence'] != target_text].copy()
        
        if len(available_samples) < k:
            print(f"警告: 可用样本数({len(available_samples)})少于请求的k值({k})")
            k = len(available_samples)
        
        # 随机采样
        sampled_examples = available_samples.sample(n=k, random_state=None).to_dict('records')
        
        return sampled_examples
    
    def create_in_context_prompt(self, target_text: str, include_reasoning: bool = True) -> str:
        """
        为目标文本创建包含in-context learning示例的完整prompt
        
        Args:
            target_text: 需要预测的目标文本
            include_reasoning: 是否包含推理过程
            
        Returns:
            完整的prompt字符串
        """
        # 采样示例
        examples = self.sample_examples(target_text)
        
        # 构建示例prompts
        example_prompts = []
        for example in examples:
            reasoning = example.get('reason', '') if include_reasoning else None
            example_prompt = self.make_one_shot_prompt(
                text=example['sentence'],
                reasoning=reasoning,
                label=example['label_name']
            )
            example_prompts.append(example_prompt)
        
        # 构建目标prompt（不包含标签）
        target_prompt = self.make_one_shot_prompt(
            text=target_text,
            reasoning=None,
            label=None
        )
        
        # 拼接所有prompts
        all_prompts = example_prompts + [target_prompt]
        full_prompt = self.make_long_shot_prompt(all_prompts)
        
        return full_prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        使用模型生成回复
        
        Args:
            prompt: 输入prompt
            
        Returns:
            模型生成的回复
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始prompt，只保留生成的部分
        response = response[len(prompt):].strip()
        
        return response
    
    def parse_prediction(self, response: str) -> Dict[str, str]:
        """
        解析模型回复，提取预测的标签
        
        Args:
            response: 模型回复
            
        Returns:
            包含predicted_label的字典
        """
        # 获取所有可能的标签名称
        valid_labels = list(self.label_mapping.values())
        
        # 提取标签
        predicted_label = None
        
        # 方案1: 寻找 "Label: xxx" 格式
        for label_name in valid_labels:
            if re.search(rf'Label:\s*{re.escape(label_name)}', response, re.IGNORECASE):
                predicted_label = label_name
                break
        
        # 方案2: 如果没找到，在整个回复中寻找标签名称
        if predicted_label is None:
            for label_name in valid_labels:
                if label_name.lower() in response.lower():
                    predicted_label = label_name
                    break
        
        # 方案3: 寻找可能的标签关键词（更宽泛的匹配）
        if predicted_label is None:
            label_keywords = {
                'positive': ['positive', 'pos', 'good', 'bullish', 'optimistic'],
                'negative': ['negative', 'neg', 'bad', 'bearish', 'pessimistic'],
                'neutral': ['neutral', 'neu', 'mixed', 'balanced']
            }
            
            response_lower = response.lower()
            for label_name in valid_labels:
                if label_name.lower() in label_keywords:
                    keywords = label_keywords[label_name.lower()]
                    if any(keyword in response_lower for keyword in keywords):
                        predicted_label = label_name
                        break
        
        return {
            'predicted_label': predicted_label or "unknown"
        }

    def predict_single(self, text: str, include_reasoning: bool = False) -> Dict[str, str]:
        """
        预测单个文本的情感标签
        
        Args:
            text: 输入文本
            include_reasoning: 是否在示例中包含推理过程（默认False，因为预测时不需要reasoning）
            
        Returns:
            预测结果字典
        """
        prompt = self.create_in_context_prompt(text, include_reasoning)
        print(f"Prompt: {prompt}")
        print("="*50)
        response = self.generate_response(prompt)
        result = self.parse_prediction(response)
        
        return {
            'text': text,
            'predicted_label': result['predicted_label'],
            'full_response': response
        }

    def evaluate_on_test_set(self, include_reasoning: bool = True, max_samples: int = None) -> pd.DataFrame:
        """
        在测试集上进行评估
        
        Args:
            include_reasoning: 是否在示例中包含推理过程
            max_samples: 最大测试样本数，None表示测试全部
            
        Returns:
            包含预测结果的DataFrame
        """
        test_data = self.test_df.copy()
        if max_samples is not None:
            test_data = test_data.head(max_samples)
        
        results = []
        
        for idx, row in test_data.iterrows():
            print(f"处理测试样本 {idx+1}/{len(test_data)}...")
            
            prediction = self.predict_single(row['sentence'], include_reasoning)
            
            result = {
                'sentence': row['sentence'],
                'true_label': row['label_name'],
                'predicted_label': prediction['predicted_label'],
                'correct': prediction['predicted_label'] == row['label_name'],
                'full_response': prediction['full_response']
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # 计算准确率
        accuracy = results_df['correct'].mean()
        print(f"\n准确率: {accuracy:.4f} ({results_df['correct'].sum()}/{len(results_df)})")
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        分析预测结果
        
        Args:
            results_df: 预测结果DataFrame
            
        Returns:
            分析统计结果
        """
        total_samples = len(results_df)
        correct_predictions = results_df['correct'].sum()
        accuracy = correct_predictions / total_samples
        
        # 按标签统计
        label_stats = {}
        for label in results_df['true_label'].unique():
            label_data = results_df[results_df['true_label'] == label]
            label_accuracy = label_data['correct'].mean()
            label_stats[label] = {
                'total': len(label_data),
                'correct': label_data['correct'].sum(),
                'accuracy': label_accuracy
            }
        
        analysis = {
            'overall_accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'label_statistics': label_stats
        }
        
        return analysis

def main():
    """
    主函数：演示in-context learning情感分类
    """
    # 配置参数
    MODEL_PATH = "/share/wangzifei03/Qwen3-4B/"
    K = 3  # in-context learning示例个数
    MAX_TEST_SAMPLES = 50  # 测试样本数（设置较小值用于快速测试）
    
    # 加载数据集
    dataset = data_loader.FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,
        random_state=42
    )
    TEXT_PREFIX = dataset.text_prefix
    LABEL_PREFIX = dataset.label_prefix
    _, test_df = dataset.get_datasets_with_label_names()
    label_mapping = dataset.get_label_mapping()
    train_df = pd.read_csv('sentiment_analysis_results_if_check_True.csv')
    # 初始化分类器
    print("初始化In-Context Learning分类器...")
    classifier = InContextLearningClassifier(
        train_df=train_df,
        test_df=test_df,
        label_mapping=label_mapping,
        text_prefix=TEXT_PREFIX,
        label_prefix=LABEL_PREFIX,
        model_path=MODEL_PATH,
        k=K,
        random_state=42
    )
    
    # 单个样本预测示例
    print("\n=== 单个样本预测示例 ===")
    test_text = "The company's quarterly earnings exceeded expectations significantly."
    result = classifier.predict_single(test_text, include_reasoning=True)
    
    print(f"输入文本: {result['text']}")
    print(f"预测标签: {result['predicted_label']}")
    
    # 测试集评估
    print("\n=== 测试集评估 ===")
    results_df = classifier.evaluate_on_test_set(
        include_reasoning=True,
        max_samples=MAX_TEST_SAMPLES
    )
    # 导出为CSV文件
    output_filename = f"in_context_learning_results_k{K}_samples{MAX_TEST_SAMPLES}.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"结果已保存到: {output_filename}")

    # 分析结果
    analysis = classifier.analyze_results(results_df)
    
    print("\n=== 详细分析结果 ===")
    print(f"总体准确率: {analysis['overall_accuracy']:.4f}")
    print(f"总样本数: {analysis['total_samples']}")
    print(f"正确预测数: {analysis['correct_predictions']}")
    
    print("\n各标签准确率:")
    for label, stats in analysis['label_statistics'].items():
        print(f"  {label}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    # 保存结果
    output_file = f"in_context_learning_results_k{K}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_file}")
    
    # 显示一些预测示例
    print("\n=== 预测示例 ===")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        status = "✓" if row['correct'] else "✗"
        print(f"{status} 真实: {row['true_label']} | 预测: {row['predicted_label']}")
        print(f"   文本: {row['sentence'][:100]}...")
        print(f"   推理: {row['reasoning'][:150]}...\n")

if __name__ == "__main__":
    main()