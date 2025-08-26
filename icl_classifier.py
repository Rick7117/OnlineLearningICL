import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Any
import re
from llm_batch_processor import BaseModelResponseGenerator
from CONSTANT import SHOT_SEPARATOR, COT_PREFIX
import data_loader

class InContextLearningClassifier(BaseModelResponseGenerator):
    """
    基于上下文学习的文本分类器，继承自BaseModelResponseGenerator
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 label_mapping: Dict, text_prefix: str, label_prefix: str, 
                 model_path: str, k: int = 3, random_state: int = 42):
        """
        初始化上下文学习分类器
        
        Args:
            train_df: 训练数据集
            test_df: 测试数据集
            label_mapping: 标签映射字典
            text_prefix: 文本前缀
            label_prefix: 标签前缀
            model_path: 模型路径
            k: 上下文示例数量
            random_state: 随机种子
        """
        super().__init__(model_path)
        self.train_df = train_df
        self.test_df = test_df
        self.label_mapping = label_mapping
        self.text_prefix = text_prefix
        self.label_prefix = label_prefix
        self.k = k
        self.random_state = random_state
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 获取所有可能的标签
        self.all_labels = list(label_mapping.values())
    
    def create_prompt(self, **kwargs) -> Dict[str, str]:
        """
        实现BaseModelResponseGenerator的抽象方法
        为in-context learning创建prompt
        
        Args:
            **kwargs: 包含sentence, include_reasoning等参数
            
        Returns:
            Dict: 包含system_prompt和user_prompt的字典
        """
        sentence = kwargs.get('sentence', '')
        include_reasoning = kwargs.get('include_reasoning', True)
        
        # 创建in-context learning prompt
        full_prompt = self.create_in_context_prompt(sentence, include_reasoning)
        user_prompt = full_prompt + self.label_prefix
        
        return {
            "system_prompt": "",
            "user_prompt": user_prompt
        }
    
    def make_one_shot_prompt(self, text: str, reasoning: str, label: str) -> str:
        """
        制作单个示例的prompt
        
        Args:
            text: 输入文本
            reasoning: 推理过程（可选）
            label: 标签（可选，推理时可能为空）
            
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
        available_samples = self.train_df[self.train_df['text'] != target_text].copy()
        
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
            reasoning = example.get('reasoning', '') if include_reasoning else None
            example_prompt = self.make_one_shot_prompt(
                text=example['text'],
                reasoning=reasoning,
                label=self.label_mapping[example['true_label']]
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
    
    def parse_response(self, response: str) -> str:
        """
        解析模型回答，提取预测的标签
        
        Args:
            response: 模型的原始回答
            
        Returns:
            预测的标签
        """
        if not response or response.strip() == "":
            return "unknown"
        
        response = response.strip()
        
        # 方案1: 查找"Label: xxx"格式
        label_pattern = r'Label:\s*([^\n\r]+)'
        match = re.search(label_pattern, response, re.IGNORECASE)
        if match:
            predicted_label = match.group(1).strip()
            # 检查是否为有效标签
            for label in self.all_labels:
                if label.lower() in predicted_label.lower():
                    return label
        
        # 方案2: 直接在回答中查找标签名
        for label in self.all_labels:
            if label.lower() in response.lower():
                return label
        
        # 方案3: 查找关键词
        positive_keywords = ['positive', 'good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_keywords = ['negative', 'bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        response_lower = response.lower()
        
        if any(keyword in response_lower for keyword in positive_keywords):
            return 'positive'
        elif any(keyword in response_lower for keyword in negative_keywords):
            return 'negative'
        
        # 如果都没找到，返回unknown
        return "unknown"
    
    def predict_batch(self, texts: List[str], include_reasoning: bool = False, 
                     batch_size: int = 4) -> List[Dict[str, str]]:
        """
        批量预测文本的情感标签
        
        Args:
            texts: 输入文本列表
            include_reasoning: 是否在示例中包含推理过程
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        batch_data = [{'sentence': text, 'include_reasoning': include_reasoning} for text in texts]
        
        # 使用父类的批量处理方法
        raw_responses = self.query_batch(batch_data, batch_size)
        
        # 解析结果
        results = []
        for text, response in zip(texts, raw_responses):
            predicted_label = self.parse_response(response)
            
            results.append({
                'text': text,
                'predicted_label': predicted_label,
                'full_response': response
            })
        
        return results
    
    def evaluate_on_test_set(self, include_reasoning: bool = True, 
                           max_samples: int = None, batch_size: int = 4) -> Dict[str, Any]:
        """
        在测试集上评估模型性能
        
        Args:
            include_reasoning: 是否包含推理过程
            max_samples: 最大测试样本数，None表示使用全部
            batch_size: 批次大小
            
        Returns:
            包含评估结果的字典
        """
        # 准备测试数据
        test_data = self.test_df.copy()
        if max_samples is not None:
            test_data = test_data.head(max_samples)
        
        texts = test_data['sentence'].tolist()
        true_labels = test_data['label_name'].tolist()
        
        print(f"开始评估，共{len(texts)}个样本...")
        
        # 批量预测
        predictions = self.predict_batch(texts, include_reasoning, batch_size)
        
        # 计算准确率
        correct = 0
        total = len(predictions)
        detailed_results = []
        
        for i, pred in enumerate(predictions):
            true_label = true_labels[i]
            predicted_label = pred['predicted_label']
            is_correct = (predicted_label == true_label)
            
            if is_correct:
                correct += 1
            
            detailed_results.append({
                'text': pred['text'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': is_correct,
                'full_response': pred['full_response']
            })
        
        accuracy = correct / total if total > 0 else 0
        
        # 计算每个标签的准确率
        label_stats = {}
        for label in self.all_labels:
            label_true = [r for r in detailed_results if r['true_label'] == label]
            label_correct = [r for r in label_true if r['correct']]
            
            label_stats[label] = {
                'total': len(label_true),
                'correct': len(label_correct),
                'accuracy': len(label_correct) / len(label_true) if len(label_true) > 0 else 0
            }
        
        results = {
            'overall_accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'label_statistics': label_stats,
            'detailed_results': detailed_results
        }
        
        # 保存详细结果到CSV文件
        results_df = pd.DataFrame(detailed_results)
        csv_filename = f"results/icl_evaluation_results_{include_reasoning}_k{self.k}_samples.csv"
        results_df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"详细结果已保存到: {csv_filename}")
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """
        打印评估结果摘要
        
        Args:
            results: evaluate_on_test_set返回的结果字典
        """
        print("\n=== 评估结果摘要 ===")
        print(f"总体准确率: {results['overall_accuracy']:.4f}")
        print(f"正确预测: {results['correct_predictions']}/{results['total_predictions']}")
        
        print("\n=== 各标签准确率 ===")
        for label, stats in results['label_statistics'].items():
            print(f"{label}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        print("\n=== 预测示例 ===")
        detailed = results['detailed_results']
        for i in range(min(5, len(detailed))):
            result = detailed[i]
            status = "✓" if result['correct'] else "✗"
            print(f"{status} 真实: {result['true_label']} | 预测: {result['predicted_label']}")
            print(f"   文本: {result['text'][:100]}...")
            print(f"   回答: {result['full_response'][:100]}...\n")

def main():
    """
    主函数，演示上下文学习分类器的使用
    """
    # 这里需要根据实际情况导入数据和设置参数
    print("上下文学习分类器演示")
    dataset = data_loader.FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,
        random_state=42
    )
    _, test_df = dataset.get_datasets_with_label_names()
    train_df = pd.read_csv('/home/huangbinbin03/longicl/data/cot_online_learning_results.csv')
    text_prefix = dataset.text_prefix
    label_prefix = dataset.label_prefix
    label_mapping = dataset.get_label_mapping()

    model_path = "/share/wangzifei03/Qwen3-4B/"
    
    classifier = InContextLearningClassifier(
        train_df=train_df,
        test_df=test_df,
        label_mapping=label_mapping,
        text_prefix=text_prefix,
        label_prefix=label_prefix,
        model_path=model_path,
        k=1
    )
    
    # 评估模型
    results = classifier.evaluate_on_test_set(
        include_reasoning=True,
        batch_size=4
    )
    
    # 打印结果
    classifier.print_evaluation_summary(results)

if __name__ == "__main__":
    main()