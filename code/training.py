import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import yaml
import numpy as np
import matplotlib.pyplot as plt


def convert_dates(date_series):
    # 定义转换日期时间的函数
    return pd.to_datetime(date_series, format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 数据路径和文件
data_path = "xlsxGL2\\"
training_files = [
    "040GL2.xlsx",
    "041GL2.xlsx",
    "042GL2.xlsx",
    "043GL2.xlsx",
    "044GL2.xlsx",]

test_file = "\\G02\\45G02L2.xlsx"

# 加载训练数据
train_data_parts = []
for file in training_files:
    temp_df = pd.read_excel(data_path + file, engine='openpyxl')
    temp_df['TIME(GPST)'] = convert_dates(temp_df['TIME(GPST)'])
    train_data_parts.append(temp_df)
train_data = pd.concat(train_data_parts, ignore_index=True)

# 加载测试数据
test_data = pd.read_excel(test_file, engine='openpyxl')
test_data['TIME(GPST)'] = convert_dates(test_data['TIME(GPST)'])

# 数据预处理
## 计算相对时间秒
train_data['TIME(GPST)_seconds'] = (train_data['TIME(GPST)'] - train_data['TIME(GPST)'].min()).dt.total_seconds()
test_data['TIME(GPST)_seconds'] = (test_data['TIME(GPST)'] - test_data['TIME(GPST)'].min()).dt.total_seconds()

## 选择特征和目标列
X_train = train_data[['TIME(GPST)_seconds', 'AZ(deg)', 'EL(deg)', 'SNR(dBHz)']]
y_train = train_data['L2 MP(m)']
X_test = test_data[['TIME(GPST)_seconds', 'AZ(deg)', 'EL(deg)', 'SNR(dBHz)']]
y_test = test_data['L2 MP(m)']

# 创建XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=40, learning_rate=0.1, max_depth=20)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# 创建包含预测结果的数据框
predictions_df = pd.DataFrame({
    'TIME(GPST)_seconds': X_test['TIME(GPST)_seconds'],
    'EL(deg)': X_test['EL(deg)'],
    'AZ(deg)': X_test['AZ(deg)'],
    'Predicted_MP(m)': y_pred
})

# 保存数据框到 Excel 文件
predictions_df.to_excel('.\\G02\\pre_G02_L1.xlsx', index=False)

# 将模型的参数和特征重要性保存为 YAML 文件
model_params = model.get_params()
feature_importances = model.feature_importances_

# 将特征重要性转换为字典，并且使值更易读
feature_importance_dict = {feature: float(importance) for feature, importance in zip(X_train.columns, feature_importances)}

# 创建一个字典来存储模型信息
model_info = {
    'model_params': {key: float(value) if isinstance(value, (np.floating, np.integer)) else value for key, value in model_params.items()},
    'feature_importances': feature_importance_dict,
    'rmse': float(rmse)
}

# 将模型信息保存为 YAML 文件
with open('..\\xgboost_model_info.yml', 'w') as file:
    yaml.dump(model_info, file, default_flow_style=False, allow_unicode=True)

print("模型信息已保存为 YAML 文件")

# 检查数据
print("Test Data (TIME(GPST)_seconds, L2 MP(m), Predicted MP(m)):")
print(pd.DataFrame({
    'TIME(GPST)_seconds': X_test['TIME(GPST)_seconds'],
    'L2 MP(m)': y_test,
    'Predicted MP(m)': y_pred
}).head())

# 绘制预测结果和测试数据集随着时间的变化
plt.figure(figsize=(10, 6))
plt.plot( y_test, label='True MP(m)', color='blue')
plt.plot( y_pred, label='Predicted MP(m)', color='red', linestyle='--')
plt.xlabel('EPOCH/s')
plt.ylabel('MP/m')
plt.title('G02 L2')
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('..\\image\\L2\\45G02L2.png')
plt.show()