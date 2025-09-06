import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('rf.pkl')

# 特征范围定义
feature_ranges = {
    "LC": {"type": "numerical", "min": 0.000, "max": 100.000, "default": 0.32},
    "WBC": {"type": "numerical", "min": 0.000, "max": 100.000, "default": 5.82},
    "Fg": {"type": "numerical", "min": 0, "max": 1000, "default": 2},
    "PLT": {"type": "numerical", "min": 0, "max": 1000, "default": 100},
    "DD": {"type": "numerical", "min": 0, "max": 42, "default": 37},
    "Age": {"type": "numerical", "min": 0, "max": 200, "default": 80},
    "PCT": {"type": "numerical", "min": 0, "max": 100, "default": 20},
    "Lung": {"type": "categorical", "options": [0, 1], "default": 0},
    "Immunocompromised": {"type": "categorical", "options": [0, 1], "default": 0}
}

# 单位信息（按需要调整）
feature_units = {
    "LC": "×10^9/L",
    "WBC": "×10^9/L",
    "Fg": "g/L",
    "PLT": "×10^9/L",
    "DD": "μg/ml",
    "Age": "years",
    "PCT": "ng/ml",
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

st.header("Enter the following feature values:")
feature_values = []
display_values = {}

# 动态生成输入项
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        unit = feature_units.get(feature, "")
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']}) {unit}",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
        display_values[feature] = f"{value} {unit}" if unit else value
    elif properties["type"] == "categorical":
        option = st.selectbox(
            label=f"{feature} (Select)",
            options=["NO", "YES"],
            index=properties["default"],
        )
        value = 1 if option == "YES" else 0  # 转换成 0/1 给模型
        display_values[feature] = option
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of PL patients is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

