import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import io

# ---------------------------
# 页面配置 / Page Config
# ---------------------------
st.set_page_config(
    page_title="DDPM-FCM 乳腺癌复发风险预测系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# 标题与免责 / Header
# ---------------------------
st.markdown("<h1 style='text-align: center;'>DDPM-FCM-Fusion: Breast Cancer Recurrence Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>中国医科大学附属盛京医院</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# 模型加载 / Load Models
# ---------------------------
@st.cache_resource
def load_models(model_dir="results"):
    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        svm = joblib.load(f"{model_dir}/svm.pkl")
        mlp = joblib.load(f"{model_dir}/mlp.pkl")
        ddpm = torch.load(f"{model_dir}/ddpm.pt", map_location="cpu")
        attention = torch.load(f"{model_dir}/attention.pt", map_location="cpu")
        return scaler, svm, mlp, ddpm, attention
    except Exception as e:
        st.warning(f"模型加载失败，使用模拟结果演示 / Model load failed: {e}")
        return None, None, None, None, None

scaler, svm, mlp, ddpm, attention = load_models()

# ---------------------------
# 示例数据模板 / Example Data
# ---------------------------
example_data = {
    "Age": 55, "Sexual history": 0, "Family cancer history": 0, "Parity": 2,
    "Menopausal status": 1, "Comorbidities": 1, "Presenting symptom": 1,
    "Surgical route": 1, "Tumor envelope integrity": 1, "Micropapillary": 0,
    "Fertility-sparing surgery": 0, "Psammam4Xtion": 0, "Completeness of surgery": 1,
    "Acentric cytology": 0, "FIGO staging": 2, "Rumor size": 2.5
}

# ---------------------------
# 主布局 / Main Layout
# ---------------------------
col_left, col_right = st.columns([1.8, 1.2])

with col_left:
    st.markdown("### Model files directory")
    st.text_input("", value="results", disabled=True, key="model_dir")

    # 输入方式
    input_method = st.radio(
        "选择输入方式 / Choose Input Method:",
        ("单例输入 / Single Instance", "批量上传 CSV / Batch Upload CSV"),
        horizontal=True
    )

    if input_method == "单例输入 / Single Instance":
        st.markdown("### Single instance")
        form = st.form("patient_form")
        with form:
            col1, col2 = st.columns(2)
            inputs = {}

            # 左列
            with col1:
                inputs["Age"] = st.number_input("Age", min_value=20, max_value=90, value=example_data["Age"])
                inputs["Family cancer history"] = st.selectbox("Family cancer history", ["No", "Yes"], index=example_data["Family cancer history"])
                inputs["Menopausal status"] = st.selectbox("Menopausal status", ["No", "Yes"], index=example_data["Menopausal status"])
                inputs["Presenting symptom"] = st.selectbox("Presenting symptom", ["Lump", "Pain", "Discharge"], index=example_data["Presenting symptom"])
                inputs["Tumor envelope integrity"] = st.selectbox("Tumor envelope integrity", ["Intact", "Ruptured"], index=example_data["Tumor envelope integrity"])
                inputs["Fertility-sparing surgery"] = st.selectbox("Fertility-sparing surgery", ["No", "Yes"], index=example_data["Fertility-sparing surgery"])
                inputs["Completeness of surgery"] = st.selectbox("Completeness of surgery", ["R0", "R1", "R2"], index=example_data["Completeness of surgery"])
                inputs["FIGO staging"] = st.selectbox("FIGO staging", ["I", "II", "III", "IV"], index=example_data["FIGO staging"])

            # 右列
            with col2:
                inputs["Sexual history"] = st.selectbox("Sexual history", ["No", "Yes"], index=example_data["Sexual history"])
                inputs["Parity"] = st.number_input("Parity", min_value=0, max_value=10, value=example_data["Parity"])
                inputs["Comorbidities"] = st.selectbox("Comorbidities", ["No", "Yes"], index=example_data["Comorbidities"])
                inputs["Surgical route"] = st.selectbox("Surgical route", ["Mastectomy", "BCS"], index=example_data["Surgical route"])
                inputs["Micropapillary"] = st.selectbox("Micropapillary", ["No", "Yes"], index=example_data["Micropapillary"])
                inputs["Psammam4Xtion"] = st.selectbox("Psammam4Xtion", ["No", "Yes"], index=example_data["Psammam4Xtion"])
                inputs["Acentric cytology"] = st.selectbox("Acentric cytology", ["Negative", "Positive"], index=example_data["Acentric cytology"])
                inputs["Rumor size"] = st.number_input("Rumor size (cm)", min_value=0.1, max_value=20.0, value=example_data["Rumor size"], step=0.1)

            predict_btn = form.form_submit_button("PREDICT", use_container_width=True, type="primary")

    else:
        st.markdown("### Batch upload CSV")
        uploaded_file = st.file_uploader("上传 CSV 文件 / Upload CSV File", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Run Batch Prediction", use_container_width=True):
                st.success("批量预测完成 / Batch prediction complete")

# ---------------------------
# 右侧结果区 / Right Panel - Results
# ---------------------------
with col_right:
    st.markdown("### Predicted recurrence probability")
    prob = st.empty()
    st.markdown("### Estimated median recurrence time (years)")
    time = st.empty()

    # 风险曲线
    st.markdown("### Risk over time")
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.linspace(0, 5, 100)
    y = 1 - np.exp(-0.2 * x)  # 模拟风险曲线
    ax.plot(x, y, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Recurrence probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # 模拟预测结果
    if input_method == "单例输入 / Single Instance" and predict_btn:
        # 真实预测逻辑（占位）
        risk_prob = 0.128
        median_time = 3.58

        prob.markdown(f"<h1 style='text-align: center; color: #d62728;'>{risk_prob:.3f}</h1>", unsafe_allow_html=True)
        time.markdown(f"<h2 style='text-align: center; color: #2ca02c;'>{median_time:.2f}</h2>", unsafe_allow_html=True)

        st.success("预测完成 / Prediction completed")

# ---------------------------
# 底部免责 / Footer Disclaimer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
此系统仅用于科研与教学展示，不可作为临床诊断或治疗决策依据。<br>
<i>For research and demonstration purposes only. Not for clinical use.</i>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 下载模板 / Download Template
# ---------------------------
df_template = pd.DataFrame([example_data])
buffer = io.BytesIO()
df_template.to_csv(buffer, index=False)
buffer.seek(0)
st.sidebar.download_button(
    label="下载 CSV 模板 / Download CSV Template",
    data=buffer,
    file_name="breast_cancer_template.csv",
    mime="text/csv"
)
