# OnlineLearningICL

在线学习的上下文学习（In-Context Learning）项目，通过在线学习机制提升示例演示的质量。

## 项目结构

### 核心模块

- **`data_loader.py`** - 数据加载器，支持Financial PhraseBank数据集的加载、预处理和划分
- **`icl_classifier.py`** - 上下文学习分类器，实现基于示例的文本分类和批量评估
- **`online_learning_cot.py`** - 在线学习CoT（Chain-of-Thought）推理生成器，核心算法实现
- **`llm_batch_processor.py`** - LLM批处理器，提供批量文本处理和情感分析功能
- **`utils.py`** - 工具函数，包含情感分析和推理生成的核心逻辑，暂时用不上后面会重构

### 配置和可视化

- **`CONSTANT.py`** - 项目常量定义（分隔符、CoT前缀等）
- **`plot.py`** - 结果可视化工具，生成准确率对比图表

### 数据和结果

- **`data/`** - 存储在线学习实验结果的CSV文件和数据集
- **`results/`** - 存储不同参数配置下的ICL评估结果
- **`accuracy_history.csv`** - 准确率历史记录
- **`accuracy_vs_k_plot.png`** - 准确率对比可视化图表

## 主要功能

1. **在线学习机制** - 通过动态更新示例的推理链提升分类性能
2. **Chain-of-Thought推理** - 集成推理过程提高预测准确性
3. **批量处理** - 支持大规模文本数据的高效处理
4. **结果可视化** - 提供详细的性能分析和对比图表
5. **多参数实验** - 支持不同k值和推理模式的对比实验

## 使用方法
1. 修改模型路径
   - 在 online_learning_cot.py 和 lm_batch_processor.py中修改`MODEL_PATH`为你的本地模型路径
2. 运行主要实验
   ```bash
   python online_learning_cot.py  # 在线学习实验
   python icl_classifier.py       # ICL分类评估
   python plot.py                 # 生成可视化图表
   ```
3. 查看结果
   - 在线学习实验结果将保存到`data/`目录下
   - ICL分类评估结果将保存到`results/`目录下
   - 可视化图表将保存为`accuracy_vs_k_plot.png`


