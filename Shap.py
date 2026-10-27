import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
import os

import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

_available_fonts = {f.name for f in fm.fontManager.ttflist}
if 'Arial' not in _available_fonts:
    print("警告：系统未检测到 Arial 字体，将回退到默认字体（可能不是 Arial）。")


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

OUTPUT_DIR = 'shap'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 重新加载数据并切分
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


# 加载已经调优好的模型

try:
    best_rf = joblib.load('best_rf_model.pkl')
    best_xgb = joblib.load('best_xgb_model.pkl')
    best_svm = joblib.load('best_svm_model.pkl')
except FileNotFoundError as e:
    print("错误：找不到模型文件！请确保你已经在之前的代码中运行了 joblib.dump 保存了模型。")
    raise e


def force_arial(fig=None):
    if fig is None:
        fig = plt.gcf()

    # 坐标轴标题、标签、刻度
    for ax in fig.axes:
        # 标题
        if ax.title is not None:
            ax.title.set_fontname('Arial')
        # 坐标轴标签
        ax.xaxis.label.set_fontname('Arial')
        ax.yaxis.label.set_fontname('Arial')
        # 刻度标签
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        # 轴内的文本（例如 SHAP 图中色条说明、注释）
        for txt in ax.texts:
            txt.set_fontname('Arial')
        # 图例
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontname('Arial')
            if leg.get_title() is not None:
                leg.get_title().set_fontname('Arial')

    # 颜色条（SHAP summary_plot 会有 colorbar）
    for ax in fig.axes:
        for cax in [ax] + list(getattr(ax, 'figure', fig).axes):
            pass  # 已在上面的循环中覆盖

    # 颜色条上的标签和刻度
    for cax in fig.axes:
        # 判断是否是 colorbar 的轴：仅有一个刻度轴时也统一处理
        for label in cax.get_xticklabels() + cax.get_yticklabels():
            label.set_fontname('Arial')
        if cax.get_ylabel():
            cax.yaxis.label.set_fontname('Arial')
        if cax.get_xlabel():
            cax.xaxis.label.set_fontname('Arial')

    # 图上的所有 Text 对象
    for txt in fig.findobj(match=plt.Text):
        try:
            txt.set_fontname('Arial')
        except Exception:
            pass


# 定义绘图函数
def plot_and_save_shap(shap_values_list, features_data, model_name):
    for i, class_name in enumerate(class_names):

        if isinstance(shap_values_list, list):
            class_shap_values = shap_values_list[i]
        elif isinstance(shap_values_list, np.ndarray) and len(shap_values_list.shape) == 3:
            class_shap_values = shap_values_list[:, :, i]
        else:
            class_shap_values = shap_values_list[i]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(class_shap_values, features_data, show=False)

        plt.title(f'{model_name} SHAP: Predicting [{class_name}] Deposit',
                  fontsize=14, fontname='Arial')

        # 把当前图中所有文本统一改为 Arial
        force_arial(plt.gcf())

        plt.tight_layout()

        safe_class_name = str(class_name).replace('/', '_').replace(' ', '_')
        file_name = f"{model_name}_SHAP_{safe_class_name}.png"

        full_save_path = os.path.join(OUTPUT_DIR, file_name)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        plt.close()


# 执行 SHAP 分析

# 随机森林
explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf.shap_values(X_test)
plot_and_save_shap(shap_values_rf, X_test, "RandomForest")

# XGBoost
explainer_xgb = shap.TreeExplainer(best_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)
plot_and_save_shap(shap_values_xgb, X_test, "XGBoost")

# SVM
background_summary = shap.kmeans(X_train, 50)
X_test_sample = shap.sample(X_test, 300, random_state=42)
svm_predict_proba_wrapper = lambda x: best_svm.predict_proba(x)
explainer_svm = shap.KernelExplainer(svm_predict_proba_wrapper, background_summary)
shap_values_svm = explainer_svm.shap_values(X_test_sample, nsamples=100)

plot_and_save_shap(shap_values_svm, X_test_sample, "SVM")

print("基于模型的SHAP 图已输出")
