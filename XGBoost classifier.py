import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import matplotlib.font_manager as fm

_available_fonts = {f.name for f in fm.fontManager.ttflist}
if 'Arial' not in _available_fonts:
    print("警告：系统未检测到 Arial 字体，将回退到默认字体。")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# 数据加载与处理
file_path = 'data_log_ratio_scaled.xlsx'
df = pd.read_excel(file_path)
target_col = 'deposit type'

features_list = ['Mg', 'Al', 'Si', 'Ti', 'Mn', 'V', 'Cr', 'Co', 'Ni', 'Zn', 'Ga', 'Sn',
                 'V/Ti', 'Ni/Co', 'Cr/Mn', 'Ti/(Al+Si)']

X = df[features_list]
y_raw = df[target_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 模型调优
xgb_base = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(class_names),
    eval_metric='mlogloss',
    random_state=42
)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_dist,
                                   n_iter=30, cv=5, scoring='f1_weighted',
                                   n_jobs=-1, random_state=42, verbose=1)
random_search.fit(X_train, y_train, sample_weight=sample_weights)
best_xgb = random_search.best_estimator_

# 模型预测与评估
y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=class_names)

# 整理要输出的文本报告
report_text = f"""========================================
【极限梯度提升 (XGBoost) 模型评估报告】
========================================
1. 最佳超参数组合: {random_search.best_params_}
2. 交叉验证最佳 F1 分数: {random_search.best_score_:.4f}

3. 测试集核心评估指标 (加权平均):
   整体准确率 (Accuracy) : {accuracy:.4f}
   整体精确率 (Precision): {precision:.4f}
   整体召回率 (Recall)   : {recall:.4f}
   整体 F1分数 (F1-Score): {f1:.4f}
========================================
4. 各矿床类型详细评估报告:
{class_report}
========================================
"""
print(report_text)

# 将报告保存为 TXT 文件
with open('XGBoost_Evaluation_Results.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

# 混淆矩阵与重要性画图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('XGBoost: Confusion Matrix for 6 Deposit Types', fontsize=14, fontname='Arial')
plt.xlabel('Predicted Deposit Type', fontsize=12, fontname='Arial')
plt.ylabel('Actual Deposit Type', fontsize=12, fontname='Arial')
plt.xticks(rotation=45, fontname='Arial')
plt.yticks(rotation=0, fontname='Arial')
plt.tight_layout()
plt.savefig('XGBoost_confusion.png', dpi=300)
plt.show()

# 提取特征重要性
importances = best_xgb.feature_importances_
importance_df = pd.DataFrame({
    'Features': features_list,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 将特征重要性数值导出为 CSV 文件
importance_df.to_csv('XGBoost_Feature_Importance.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=importance_df, palette='Reds_r')
plt.title('XGBoost: Magnetite Feature Importance Ranking', fontsize=14, fontname='Arial')
plt.xlabel('Importance Weight (Gain)', fontsize=12, fontname='Arial')
plt.ylabel('Geochemical Features', fontsize=12, fontname='Arial')
plt.xticks(fontname='Arial')
plt.yticks(fontname='Arial')
plt.tight_layout()
plt.savefig('XGBoost_feature_importance.png', dpi=300)
plt.show()

print("代码运行完毕，调优后的SVM结果和图表已输出。")
joblib.dump(best_xgb, 'best_xgb_model.pkl')
print("XGBoost 模型已保存到本地！")

