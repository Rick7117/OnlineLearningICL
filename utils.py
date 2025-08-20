from CONSTANT import SHOT_SEPARATOR, COT_PREFIX
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, json

# def make_one_shot_prompt(text: str, reasoning: str = None, label: str = None, text_prefix: str = "", label_prefix: str = "") -> str:
#     """
#     制作prompt函数
    
#     Args:
#         text: 输入文本
#         label: 标签（可选，推理时可能为空）
#         text_prefix: 文本前缀
#         label_prefix: 标签前缀
        
#     Returns:
#         格式化的prompt字符串
#     """
#     # 构建基础部分：text_prefix + text + \n
#     prompt = text_prefix + text + "\n"
#     if reasoning is not None and reasoning != "":
#         prompt += COT_PREFIX + reasoning + "\n"
    
#     # 如果label不为空，添加label部分
#     if label is not None and label != "":
#         prompt += label_prefix + label
    
#     return prompt

# def make_long_shot_prompt(shot_prompts: list) -> str:
#     """
#     将多个one shot prompt拼接成长文本
    
#     Args:
#         shot_prompts: 包含多个one shot prompt的列表
        
#     Returns:
#         使用分隔符拼接的长文本prompt
#     """
#     if not shot_prompts:
#         return ""
    
#     # 过滤掉空的prompt
#     valid_prompts = [prompt for prompt in shot_prompts if prompt and prompt.strip()]
    
#     if not valid_prompts:
#         return ""
    
#     # 使用分隔符拼接所有prompt
#     return SHOT_SEPARATOR.join(valid_prompts)

class SentimentAndReasoningGenerator:
    def __init__(self, model_path: str):
        """
        初始化LLMs
        
        Args:
            model_path: 本地模型路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def create_cot_prompt(self, sentence: str, label_prefix: str = "sentiment", text_prefix: str = "Sentence", label_mapping: dict = None) -> str:
        """
        创建带有CoT推理的英文prompt
        
        Args:
            sentence: 输入文本
            label_prefix: 标签前缀，用于替换prompt中的"sentiment"
            text_prefix: 文本前缀，用于替换prompt中的"Sentence"
            label_mapping: 标签映射字典，格式为 {label_id: label_name}
            
        Returns:
            格式化的prompt
        """
        # 默认标签映射
        if label_mapping is None:
            label_mapping = {
                0: "negative",
                1: "neutral", 
                2: "positive"
            }
        
        # 构建标签选项字符串
        label_options = []
        for label_id, label_name in sorted(label_mapping.items()):
            label_options.append(f"{label_id} ({label_name})")
        label_choices = ", ".join(label_options)
        
        prompt = f"""Analyze the {label_prefix} of the following sentence step by step.

    {text_prefix}: "{sentence}"

    Please follow this format:
    1. Reasoning: [Explain your analysis of the {label_prefix}, considering key words, phrases, and overall tone]
    2. Label: [Choose one: {label_choices}]

    Analysis:"""
        return prompt

    def create_correction_prompt(self, sentence, reasoning, predicted_label, true_label, label_mapping):
        """
        创建纠正prompt，要求模型重新分析
        
        Args:
            sentence: 原始句子
            reasoning: 之前的推理过程
            predicted_label: 预测的标签ID
            true_label: 真实的标签ID
            label_mapping: 标签映射字典
        
        Returns:
            纠正prompt字符串
        """
        # 获取标签名称
        predicted_label_name = label_mapping.get(predicted_label, f"Unknown({predicted_label})")
        true_label_name = label_mapping.get(true_label, f"Unknown({true_label})")
        
        # 构建所有可能的标签选项
        label_options = [f"{label_id} ({label_name})" for label_id, label_name in label_mapping.items()]
        label_options_str = ", ".join(label_options)
        
        correction_prompt = f"""You previously analyzed this sentence but made an incorrect prediction. Please reconsider and provide a corrected analysis.

    Sentence: "{sentence}"

    Your previous analysis:
    Reasoning: {reasoning}
    Predicted Label: {predicted_label} ({predicted_label_name})

    However, the correct label should be: {true_label} ({true_label_name})

    Please provide a new analysis that correctly identifies the sentiment as "{true_label_name}". Consider what aspects of the sentence you might have missed or misinterpreted.

    Available labels: {label_options_str}

    Please follow this format:
    1. Reasoning: [Your step-by-step analysis explaining why this sentence expresses {true_label_name} sentiment]
    2. Label: {true_label}"""
        
        return correction_prompt
    
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """
        使用LLM生成回复
        
        Args:
            prompt: 输入prompt
            max_length: 最大生成长度
            
        Returns:
            模型生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始prompt，只保留生成的部分
        response = response[len(prompt):].strip()
        return response
    
    def parse_response(self, response: str, label_prefix: str = "sentiment", text_prefix: str = "Sentence", sentence: str = "", label_mapping: dict = None) -> tuple:
        """
        解析模型回复，提取reasoning和label
        
        Args:
            response: 模型生成的回复
            label_prefix: 标签前缀，用于过滤prompt模板
            text_prefix: 文本前缀，用于过滤prompt模板
            sentence: 原始句子，用于过滤prompt模板
            label_mapping: 标签映射字典，用于确定有效的标签值
            
        Returns:
            (reasoning, predicted_label) 元组
        """
        reasoning = ""
        predicted_label = None
        
        # 设置默认label_mapping
        if label_mapping is None:
            label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        
        # 从label_mapping中获取有效的标签ID
        valid_label_ids = list(label_mapping.keys())
        valid_label_pattern = '|'.join(map(str, valid_label_ids))
        
        try:
            # 定义需要过滤的prompt模板关键词
            filter_patterns = [
                f"Analyze the {label_prefix} of the following sentence step by step",
                f"{text_prefix}: \"{sentence}\"",
                "Please follow this format:",
                "[Explain your analysis of the",
                "considering key words, phrases, and overall tone]",
                "[Choose one:",
                "Analysis:",
            ]
            
            # 清理response，移除prompt模板内容
            cleaned_response = response
            for pattern in filter_patterns:
                # 使用正则表达式移除包含这些模式的行
                pattern_regex = re.escape(pattern).replace(r'\"', '"')
                cleaned_response = re.sub(f".*{pattern_regex}.*\n?", "", cleaned_response, flags=re.IGNORECASE)
            
            # 移除多余的空行和空白字符
            cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response).strip()
            
            # 提取所有Reasoning部分
            reasoning_matches = re.findall(
                r'(?:1\.\s*)?Reasoning:\s*(.+?)(?=(?:2\.\s*)?Label:|$)', 
                cleaned_response, 
                re.DOTALL | re.IGNORECASE
            )
            
            # 提取所有Label部分 - 使用动态的标签模式
            label_matches = re.findall(
                rf'(?:2\.\s*)?Label:\s*({valid_label_pattern})\b', 
                cleaned_response, 
                re.IGNORECASE
            )
            
            # 如果没有找到标准格式，尝试备用方案
            if not reasoning_matches:
                # 寻找任何看起来像推理的文本块
                reasoning_alt = re.findall(
                    r'(?:because|since|due to|the reason|analysis|explanation)[^.]*[.!?](?:[^.!?]*[.!?])*',
                    cleaned_response,
                    re.IGNORECASE
                )
                if reasoning_alt:
                    reasoning_matches = [' '.join(reasoning_alt)]
            
            if not label_matches:
                # 备用方案：直接搜索有效的标签数字
                number_matches = re.findall(rf'\b({valid_label_pattern})\b', cleaned_response)
                if number_matches:
                    label_matches = number_matches
            
            # 处理多个结果的情况
            if reasoning_matches:
                # 选择最长且最有意义的reasoning
                reasoning_candidates = [r.strip() for r in reasoning_matches if r.strip()]
                if reasoning_candidates:
                    # 过滤掉明显是prompt模板的内容
                    filtered_reasoning = []
                    for r in reasoning_candidates:
                        # 检查是否包含prompt模板的特征
                        is_template = any([
                            "[Explain your analysis" in r,
                            "considering key words" in r,
                            "Choose one:" in r,
                            len(r.strip()) < 10  # 太短的内容
                        ])
                        if not is_template:
                            filtered_reasoning.append(r)
                    
                    if filtered_reasoning:
                        # 选择最长的reasoning作为最终结果
                        reasoning = max(filtered_reasoning, key=len).strip()
            
            if label_matches:
                # 选择最后一个有效的label（通常是最终答案）
                valid_labels = []
                for l in label_matches:
                    try:
                        label_int = int(l)
                        if label_int in valid_label_ids:
                            valid_labels.append(label_int)
                    except ValueError:
                        continue
                
                if valid_labels:
                    predicted_label = valid_labels[-1]  # 取最后一个
            
            # 最终清理reasoning
            if reasoning:
                # 移除开头的标点符号和空白
                reasoning = re.sub(r'^[\s\-\*\•]+', '', reasoning)
                # 移除结尾多余的标点
                reasoning = re.sub(r'[\s\.]+$', '.', reasoning)
                # 确保reasoning不为空且有意义
                if len(reasoning.strip()) < 5:
                    reasoning = ""
        
        except Exception as e:
            print(f"解析回复时出错: {e}")
            print(f"原始回复: {response[:200]}...")  # 只打印前200字符避免过长
        
        return reasoning, predicted_label

    def analyze_sentiment(self, sentence: str, label_prefix: str = "sentiment", text_prefix: str = "Sentence", label_mapping: dict = None, if_reGenerate: bool = False) -> dict:
        """
        对单个句子进行情感分析
        
        Args:
            sentence: 输入文本
            label_prefix: 标签前缀
            text_prefix: 文本前缀
            label_mapping: 标签映射
            
        Returns:
            包含reasoning和predicted_label的字典
        """
        if if_reGenerate:
            prompt = self.create_correction_prompt(
                        sentence=sentence,
                        reasoning=self.reason,
                        predicted_label=self.predicted_label,
                        true_label=,
                        label_mapping=label_mapping
                    )
        else:
            prompt = self.create_cot_prompt(sentence, label_prefix, text_prefix, label_mapping)
        response = self.generate_response(prompt)
        reasoning, predicted_label = self.parse_response(response, label_prefix, text_prefix, sentence, label_mapping)
        self.reason, self.predicted_label = reasoning, predicted_label
        
        return {
            'reasoning': reasoning,
            'predicted_label': predicted_label,
            'raw_response': response
        }
        
if __name__ == "__main__":
    # 示例one shot prompts
    shot_list = [
        "文本: 这个产品很棒\n标签: positive",
        "文本: 服务态度不好\n标签: negative", 
        "文本: 还可以，一般般\n标签: neutral",
        "文本: 非常满意这次购物体验\n标签: positive"
    ]
    
    # 生成长文本prompt
    long_prompt = make_long_shot_prompt(shot_list)
    print("生成的长文本prompt:")
    print(long_prompt)
    print()
    
    # 测试空列表
    empty_result = make_long_shot_prompt([])
    print("空列表结果:", repr(empty_result))
    
    # 测试包含空字符串的列表
    mixed_list = ["文本: 好评\n标签: positive", "", "文本: 差评\n标签: negative", None]
    mixed_result = make_long_shot_prompt(mixed_list)
    print("\n过滤后的结果:")
    print(mixed_result)