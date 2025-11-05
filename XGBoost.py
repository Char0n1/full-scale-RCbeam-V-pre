import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from bayes_opt import BayesianOptimization


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
def xgb_cv(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma):
    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        random_state=42,
        n_jobs=-1
    )

    # 使用5折交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(cv_scores)

# 定义超参数搜索空间
pbounds = {
    'n_estimators': (0, 1000),
    'learning_rate': (0.001, 0.95),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5)
}

# 创建贝叶斯优化器
optimizer = BayesianOptimization(
    f=xgb_cv,
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
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

print("最佳超参数:", best_params)

# 使用最佳参数构建XGBoost回归模型
model = xgb.XGBRegressor(**best_params, random_state=42)

# 训练模型
model.fit(X_train, y_train)
# 保存最优模型
joblib.dump(model, 'xgb_best_model-200-无I.pkl')
print("最优模型已成功保存为 'xgb_best_model.pkl'")
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"测试集MSE: {mse:.4f}")
print(f"测试集R²: {r2:.4f}")
print(f"测试集MAPE: {mape:.4f}%")

# 可视化：实际值与预测值散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

# 设置相同的坐标轴范围
max_val = max(y_test.max(), y_pred.max()) * 1.1  # 留10%的边距
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# 绘制对角线
plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='完美预测线')

# 添加±30%线
plt.plot([0, max_val], [0, max_val*1.3], 'g--', lw=1.5, label='+30%误差线')
plt.plot([0, max_val], [0, max_val*0.7], 'g--', lw=1.5, label='-30%误差线')
plt.plot([0, max_val], [0, max_val*1.5], 'g--', lw=1.5, label='+30%误差线')
plt.plot([0, max_val], [0, max_val*0.5], 'g--', lw=1.5, label='-30%误差线')

plt.xlabel('实际值 (Actual V_ratio)')
plt.ylabel('预测值 (Predicted V_ratio)')
plt.title('XGBoost 预测效果散点图 (测试集)')
plt.legend()
plt.grid(True)
plt.show()

# 可选：输出特征重要性
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_columns, feature_importance)
plt.xlabel('特征重要性')
plt.title('XGBoost 特征重要性')
plt.show()
# 输出特征重要性数值
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    '特征名称': feature_columns,
    '重要性': feature_importance
}).sort_values(by='重要性', ascending=False)

print("特征重要性（从高到低）：")
print(importance_df)
