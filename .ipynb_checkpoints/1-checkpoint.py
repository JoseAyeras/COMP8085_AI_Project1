import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
file_path = '/1.csv'
df = pd.read_csv(file_path)

# 排除 'attack_cat' 和 'Label' 列（用于测试）
df_features = df.drop(columns=['attack_cat', 'Label'], errors='ignore')

# 将分类变量转换为数值格式（如果有需要）
df_features = df_features.apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)

# 计算相关性矩阵
correlation_matrix = df_features.corr()

# 使用热力图可视化相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('特征相关性矩阵')
plt.show()

# 选择与某个目标变量相关的特征，例如 'Label'（你可以根据需要更改为其他目标列）
# 假设 'Label' 存在，重新加载数据以分析与 'Label' 的相关性
df_with_label = df.drop(columns=['attack_cat'], errors='ignore')

# 将 'Label' 也转换为数值格式（如果是分类数据）
df_with_label['Label'] = pd.factorize(df_with_label['Label'])[0]

# 计算每个特征与 'Label' 的相关性
label_corr = df_with_label.corr()['Label'].sort_values(ascending=False)

# 输出与 'Label' 最相关的特征
print("与 'Label' 最相关的特征:\n", label_corr)

# 根据相关性选择最相关的几个特征
selected_features = label_corr.index[1:6]  # 选取与 'Label' 相关性最高的前5个特征
print("选取的特征:\n", selected_features)

# 使用选定的特征进行后续分析
df_selected = df[selected_features]
