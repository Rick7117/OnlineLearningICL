import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional, Union

class BaseModelResponseGenerator(ABC):
    """模型回复生成基类"""
    
    def __init__(self, model_path: str):
        """
        初始化模型和tokenizer
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left'
        )
    
    @abstractmethod
    def create_prompt(self, **kwargs) -> Dict[str, str]:
        """
        创建prompt的抽象方法，需要在子类中实现
        
        Args:
            **kwargs: 灵活的输入参数
            
        Returns:
            Dict: 包含system_prompt和user_prompt的字典
        """
        pass

    def query_batch(self, batch_data: List[Dict[str, Any]], batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        批量查询处理
        
        Args:
            batch_data: 包含每一条查询参数的字典列表
            batch_size: 批次大小
            
        Returns:
            List[Dict]: 解析后的JSON结果列表
        """
        results = []
        n = len(batch_data)
        
        for i in range(0, n, batch_size):
            batch_items = batch_data[i:i + batch_size]
            
            batch_messages = []
            for item in batch_items:
                prompt_dict = self.create_prompt(**item)
                system_prompt = prompt_dict.get("system_prompt", "你是一个AI助手。")
                user_prompt = prompt_dict["user_prompt"]
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                batch_messages.append(messages)
            
            batch_inputs = []
            for messages in batch_messages:
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                batch_inputs.append(text)
            
            model_inputs = self.tokenizer(
                batch_inputs, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            input_lengths = model_inputs.input_ids.shape[1]
            for j, generated_id in enumerate(generated_ids):
                response = self.tokenizer.decode(
                    generated_id[input_lengths:], 
                    skip_special_tokens=True
                ).strip()
            
                results.append(response)
                # try:
                #     result = json.loads(response)
                #     results.append(result)
                # except json.JSONDecodeError:
                #     print(f"JSON解析失败：{response}")
                #     results.append({"error": "解析失败"})   
        return results

    def process_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str] = None, batch_size: int = 4) -> pd.DataFrame:
        """
        处理DataFrame数据
        
        Args:
            df: 输入DataFrame
            column_mapping: 列名映射，将DataFrame列名映射到create_prompt的参数名
            batch_size: 批次大小
            
        Returns:
            pd.DataFrame: 包含结果的新DataFrame
        """
        if column_mapping is None:
            column_mapping = {}
        
        # 准备批量数据
        batch_data = []
        for _, row in df.iterrows():
            item = {}
            for df_col, param_name in column_mapping.items():
                if df_col in row:
                    item[param_name] = row[df_col]
            batch_data.append(item)
        
        results = self.query_batch(batch_data, batch_size)
        
        # 将结果添加到DataFrame中
        result_df = pd.DataFrame()
        if results and len(results) > 0:
            for key in results[0].keys():
                result_df[key] = [result.get(key, None) for result in results]
        
        return result_df
    
class SentimentAnalysisGenerator(BaseModelResponseGenerator):
    """情感分析生成器"""
    
    def create_prompt(self, **kwargs) -> Dict[str, str]:
        sentence = kwargs.get('sentence', '')
        
        system_prompt = "你是一个情感分析专家。"
        user_prompt = f"""请分析以下文本的情感倾向：\n文本: {sentence}\n请严格按照以下JSON格式输出，不要有任何额外的解释或文本：\n{{"sentiment": "正面/负面/中性"}}"""
        
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

if __name__ == "__main__":
    import data_loader
    
    # 加载数据集
    dataset = data_loader.FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,
        random_state=42
    )
    
    train_df, test_df = dataset.get_datasets_with_label_names()
    label_mapping = dataset.get_label_mapping()
    sample = train_df.head(2)
    # sample = train_df.iloc[0].to_dict()
    print(sample)
    column_mapping = {
        'sentence': 'sentence'
    }
    # 初始化在线学习分类器
    model_path = "/share/wangzifei03/Qwen3-4B/"  # 请根据实际路径修改

    analysor = SentimentAnalysisGenerator(model_path)
    print(analysor.process_dataframe(sample, column_mapping))
    
