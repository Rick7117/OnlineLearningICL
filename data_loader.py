from datasets import load_dataset
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetBase(ABC):
    """数据集抽象基类"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, text_prefix: str = 'Text:', label_prefix: str = 'Sentiment:'):
        """
        初始化数据集基类
        
        Args:
            test_size: 测试集比例，默认0.2（即8:2划分）
            random_state: 随机种子，确保结果可复现
        """
        self.test_size = test_size
        self.random_state = random_state
        self.train_df = None
        self.test_df = None
        self.text_prefix = text_prefix 
        self.label_prefix = label_prefix
    
    @abstractmethod
    def load_raw_data(self) -> Dict[str, Any]:
        """
        加载原始数据集
        
        Returns:
            原始数据集字典
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            raw_data: 原始数据
            
        Returns:
            预处理后的DataFrame
        """
        pass
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和测试集
        
        Args:
            df: 完整的数据集DataFrame
            
        Returns:
            训练集和测试集的DataFrame元组
        """
        if len(df) < 2:
            raise ValueError("数据集样本数量太少，无法进行划分")
            
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def get_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取处理好的训练集和测试集
        
        Returns:
            训练集和测试集的DataFrame元组
        """
        # 加载原始数据
        raw_data = self.load_raw_data()
        
        # 预处理数据
        processed_df = self.preprocess_data(raw_data)
        
        # 检查是否已有测试集
        if 'test' in raw_data and len(raw_data['test']) > 0:
            # 如果有测试集，分别处理
            test_raw = {'train': raw_data['test']}
            test_df = self.preprocess_data(test_raw)
            self.train_df = processed_df
            self.test_df = test_df
        else:
            # 如果没有测试集，从训练集中划分
            self.train_df, self.test_df = self.split_data(processed_df)
        
        return self.train_df, self.test_df
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            数据集信息字典
        """
        info = {}
        if self.train_df is not None:
            info['train_size'] = len(self.train_df)
            info['train_columns'] = list(self.train_df.columns)
        if self.test_df is not None:
            info['test_size'] = len(self.test_df)
            info['test_columns'] = list(self.test_df.columns)
        return info

class FinancialPhraseBankDataset(DatasetBase):
    """Financial PhraseBank数据集类"""
    
    def __init__(self, config_name: str = "sentences_allagree", test_size: float = 0.2, random_state: int = 42):
        """
        初始化Financial PhraseBank数据集
        
        Args:
            config_name: 数据集配置名称，默认"sentences_allagree"
            test_size: 测试集比例，默认0.2
            random_state: 随机种子
        """
        super().__init__(test_size, random_state)
        self.config_name = config_name
    
    def load_raw_data(self) -> Dict[str, Any]:
        """
        加载Financial PhraseBank原始数据
        
        Returns:
            原始数据集字典
        """
        try:
            dataset = load_dataset(
                "financial_phrasebank",
                self.config_name,
                trust_remote_code=True
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"加载数据集失败: {e}")
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        预处理Financial PhraseBank数据
        
        Args:
            raw_data: 原始数据集
            
        Returns:
            预处理后的DataFrame
        """
        # 获取训练数据
        train_data = raw_data['train']
        
        # 转换为DataFrame
        df = pd.DataFrame({
            'sentence': train_data['sentence'],
            'label': train_data['label']
        })
        
        # 数据清洗
        df = df.dropna()  # 删除空值
        df = df.drop_duplicates()  # 删除重复项
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        获取标签映射
        
        Returns:
            标签映射字典
        """
        # Financial PhraseBank的标签映射
        return {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
    
    def add_label_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加标签名称列
        
        Args:
            df: 原始DataFrame
            
        Returns:
            添加标签名称的DataFrame
        """
        label_mapping = self.get_label_mapping()
        df = df.copy()
        df['label_name'] = df['label'].map(label_mapping)
        return df
    
    def get_datasets_with_label_names(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取包含标签名称的数据集
        
        Returns:
            包含标签名称的训练集和测试集
        """
        train_df, test_df = self.get_datasets()
        
        train_df = self.add_label_names(train_df)
        test_df = self.add_label_names(test_df)
        
        return train_df, test_df

if __name__ == "__main__":
    # 创建数据集实例
    dataset = FinancialPhraseBankDataset(
        config_name="sentences_allagree",
        test_size=0.2,  # 8:2划分
        random_state=42
    )

    # 获取训练集和测试集
    train_df, test_df = dataset.get_datasets()

    print("训练集信息:")
    print(f"样本数量: {len(train_df)}")
    print(f"列名: {train_df.columns.tolist()}")
    print(f"标签分布:\n{train_df['label'].value_counts()}")
    print("\n前5行数据:")
    print(train_df.head())

    print("\n测试集信息:")
    print(f"样本数量: {len(test_df)}")
    print(f"列名: {test_df.columns.tolist()}")
    print(f"标签分布:\n{test_df['label'].value_counts()}")

    # 获取包含标签名称的数据集
    train_df_with_names, test_df_with_names = dataset.get_datasets_with_label_names()
    print("\n包含标签名称的训练集前5行:")
    print(train_df_with_names.head())
    print("\n包含标签名称的测试集前5行:")
    print(train_df_with_names.head())

    # 获取数据集信息
    info = dataset.get_info()
    print("\n数据集信息:")
    print(info)

    # 获取标签映射
    label_mapping = dataset.get_label_mapping()
    print("\n标签映射:")
    print(label_mapping)

    # 获取prefix 
    print("文本和标签前缀：")
    print(f"{dataset.text_prefix}\n{dataset.label_prefix}")

    # 新增：为每个标签打印一个样例
    print("\n=== 每个标签的样例 ===")
    for label_id, label_name in sorted(label_mapping.items()):
        # 从训练集中找到该标签的样例
        label_samples = train_df_with_names[train_df_with_names['label'] == label_id]
        
        if len(label_samples) > 0:
            # 随机选择一个样例
            sample = label_samples.sample(n=1, random_state=42).iloc[0]
            print(f"\n标签 {label_id} ({label_name}) 的样例:")
            print(f"  句子: {sample['sentence']}")
            print(f"  标签: {sample['label']} -> {sample['label_name']}")
        else:
            print(f"\n标签 {label_id} ({label_name}): 没有找到样例")

    # 额外：显示每个标签的统计信息
    print("\n=== 标签统计详情 ===")
    for label_id, label_name in sorted(label_mapping.items()):
        train_count = (train_df['label'] == label_id).sum()
        test_count = (test_df['label'] == label_id).sum()
        total_count = train_count + test_count
        
        print(f"标签 {label_id} ({label_name}):")
        print(f"  训练集: {train_count} 样本")
        print(f"  测试集: {test_count} 样本")
        print(f"  总计: {total_count} 样本")
        if total_count > 0:
            print(f"  训练集占比: {train_count/total_count:.2%}")
        print()