import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
import joblib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
file_path = "../抗剪承载力-分组比率_汇总-200-无I.xlsx"
df = pd.read_excel(file_path, sheet_name="抗剪承载力比例")

# 定义特征列和目标列
feature_columns = ['b_ratio', 'h0_ratio', 's_ratio', 'L0_ratio', 'Asv_ratio',
                   'fcu_ratio', 'ft_ratio', "fc'_ratio", 'fyv_ratio']
target_column = 'V_ratio'

X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义贝叶斯优化目标函数
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth > 1 else None,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=min(max_features, 0.999),  # 确保不超过1
        random_state=42,
        n_jobs=-1
    )

    # 使用5折交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(cv_scores)

# 定义超参数搜索空间
pbounds = {
    'n_estimators': (1, 1000),
    'max_depth': (1, 50),
    'min_samples_split': (2, 50),
    'min_samples_leaf': (2, 50),
    'max_features': (0.001, 0.999)
}

# 创建贝叶斯优化器
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
)

# 执行优化
optimizer.maximize(
    init_points=5,  # 初始随机探索点数
    n_iter=45,  # 贝叶斯优化迭代次数
)

# 获取最佳参数
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 1 else None
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

print("最佳超参数:", best_params)

# 使用最佳参数构建随机森林回归模型
model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

# 训练模型
model.fit(X_train, y_train)
# 保存最优模型
joblib.dump(model, '../rf_best_model-无I.pkl')
print("最优模型已成功保存为 'rf_best_model.pkl'")
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算MAPE
# 避免除以零的情况：检查y_test中是否有零，如果有零则使用np.where替换为零的项（或跳过），但这里假设没有零
# 如果y_test中有零，可以考虑使用平均绝对误差（MAE）或其他指标，或者处理数据避免零值
if (y_test == 0).any():
    print("警告：测试集中存在实际值为0的情况，MAPE可能无法计算。")
    # 可以选择跳过零值或使用其他方法
    # 这里我们使用np.where将0替换为一个很小的数，避免除以零
    y_test_no_zero = np.where(y_test == 0, 1e-10, y_test)
    mape = np.mean(np.abs((y_test_no_zero - y_pred) / y_test_no_zero)) * 100
else:
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"测试集MSE: {mse:.4f}")
print(f"测试集R²: {r2:.4f}")
print(f"测试集MAPE: {mape:.4f}%")

# 可视化：实际值与预测值散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际值 (Actual V_ratio)')
plt.ylabel('预测值 (Predicted V_ratio)')
plt.title('随机森林 预测效果散点图 (测试集)')
plt.grid(True)
plt.show()

# 可选：输出特征重要性
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_columns, feature_importance)
plt.xlabel('特征重要性')
plt.title('随机森林 特征重要性')
plt.show()

