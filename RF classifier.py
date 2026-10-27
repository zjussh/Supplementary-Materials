import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 模型调优
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5,
                           scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# 模型预测与评估
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_rf.classes_)

# 整理要输出的文本报告
report_text = f"""========================================
【随机森林 (RF) 模型评估报告】
========================================
1. 最佳超参数组合: {grid_search.best_params_}
2. 交叉验证最佳 F1 分数: {grid_search.best_score_:.4f}

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
with open('RF_Evaluation_Results.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

# 混淆矩阵与重要性画图
cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.title('RF: Confusion Matrix for 6 Deposit Types', fontsize=14, fontname='Arial')
plt.xlabel('Predicted Deposit Type', fontsize=12, fontname='Arial')
plt.ylabel('Actual Deposit Type', fontsize=12, fontname='Arial')
plt.xticks(rotation=45, fontname='Arial')
plt.yticks(rotation=0, fontname='Arial')
plt.tight_layout()
plt.savefig('RF_confusion.png', dpi=300)
plt.show()

# 计算特征重要性
importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Features': features_list,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 将特征重要性数值导出为 CSV 文件
importance_df.to_csv('RF_Feature_Importance.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=importance_df, palette='magma')
plt.title('RF: Magnetite Feature Importance Ranking', fontsize=14, fontname='Arial')
plt.xlabel('Importance Weight (Gini Importance)', fontsize=12, fontname='Arial')
plt.ylabel('Geochemical Features', fontsize=12, fontname='Arial')
plt.xticks(fontname='Arial')
plt.yticks(fontname='Arial')
plt.tight_layout()
plt.savefig('RF_feature_importance.png', dpi=300)
plt.show()

print("代码运行完毕，调优后的SVM结果和图表已输出。")

joblib.dump(best_rf, 'best_rf_model.pkl')
print("RF 模型已保存到本地！")