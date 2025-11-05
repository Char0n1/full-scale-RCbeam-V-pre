import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os

# 创建保存图片的文件夹
plot_dir = "SHAP_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 1. 读取数据
df = pd.read_excel("E:\python\intermental learning\shear\尺寸效应\抗剪承载力-分组比率_汇总-57-无I.xlsx", sheet_name="抗剪承载力比例")

# 2. 数据预处理

# 定义特征列和目标列
feature_columns = ['b_ratio', 'h0_ratio', 's_ratio', 'L0_ratio', 'Asv_ratio',
                   'fcu_ratio', 'ft_ratio', "fc'_ratio", 'fyv_ratio']
target_column = 'V_ratio'

X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 4. SHAP 分析
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)  # 这里改为直接调用explainer，返回Explanation对象

# 5. 绘制 SHAP 摘要图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "SHAP_summary_plot.png"), dpi=300, bbox_inches='tight')
plt.show()

# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 0  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 1  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 2  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 3  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 4  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 5  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 6  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()
# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 7  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()

# 6. 绘制 SHAP 依赖图（以第一个特征为例）
feature_idx = 8  # 可替换为其他特征索引
plt.figure(figsize=(10, 6))
shap.dependence_plot(feature_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Dependence Plot for {X.columns[feature_idx]}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_dependence_plot_{X.columns[feature_idx]}.png"), dpi=300, bbox_inches='tight')
plt.show()






# 7. 绘制 SHAP 力图（以第一个样本为例）
sample_idx = 0
plt.figure(figsize=(12, 4))
shap.force_plot(explainer.expected_value, shap_values.values[sample_idx, :], X_test.iloc[sample_idx, :],
                feature_names=X.columns, matplotlib=True, show=False)
plt.title(f"SHAP Force Plot for Sample {sample_idx}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"SHAP_force_plot_sample_{sample_idx}.png"), dpi=300, bbox_inches='tight')
plt.show()

# 8. 绘制 SHAP 交互作用摘要图
try:
    # 计算交互值
    shap_interaction_values = explainer.shap_interaction_values(X_test)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_interaction_values, X_test, feature_names=X.columns, show=False)
    plt.title("SHAP Interaction Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "SHAP_interaction_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"交互作用图生成失败: {e}")

# 9. 绘制 SHAP 热图 - 修复版本
plt.figure(figsize=(12, 8))
# 使用Explanation对象来绘制热图
shap.plots.heatmap(shap_values, show=False)
plt.title("SHAP Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "SHAP_heatmap.png"), dpi=300, bbox_inches='tight')
plt.show()

# 10. 输出特征重要性排序
shap_sum = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': shap_sum
}).sort_values(by='shap_importance', ascending=False)
print("Feature Importance based on SHAP:")
print(importance_df)

# 11. 额外的可视化：条形图显示特征重要性
plt.figure(figsize=(10, 6))
importance_df.plot(kind='barh', x='feature', y='shap_importance', legend=False)
plt.title('Feature Importance based on SHAP values')
plt.xlabel('mean(|SHAP value|)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "SHAP_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"所有图片已保存到 {plot_dir} 文件夹中")