import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import io

# ---------------------------
# 应用标题与署名 / App Header
# ---------------------------
st.set_page_config(page_title="DDPM-FCM 乳腺癌复发预测系统 / Breast Cancer Recurrence Prediction System", layout="wide")
st.title("🩺 DDPM-FCM 乳腺癌复发风险评估 / Breast Cancer Recurrence Risk Evaluation")
st.markdown("""
**中国医科大学附属盛京医院 Shengjing Hospital of China Medical University**  
本系统仅供科研与教学演示使用，不能作为临床诊断依据。  
_This system is for research and educational demonstration only, not for clinical decision-making._
""")

# ---------------------------
# 模型加载 / Load Models
# ---------------------------
def load_models(model_dir="results"):
    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        svm = joblib.load(f"{model_dir}/svm.pkl")
        mlp = joblib.load(f"{model_dir}/mlp.pkl")
        ddpm, attention = None, None
        try:
            ddpm = torch.load(f"{model_dir}/ddpm.pt", map_location="cpu")
            attention = torch.load(f"{model_dir}/attention.pt", map_location="cpu")
        except:
            st.warning("未检测到 DDPM/Attention 模型文件，使用基本特征预测 / DDPM/Attention not found, using basic features.")
        return scaler, svm, mlp, ddpm, attention
    except Exception as e:
        st.error(f"模型加载失败 / Model loading failed: {e}")
        return None, None, None, None, None

scaler, svm, mlp, ddpm, attention = load_models()

# ---------------------------
# 示例 CSV 模板 / Sample CSV Template
# ---------------------------
example_data = {
    "Age": [52],
    "Tumor_Size_mm": [25],
    "Lymph_Node_Positive": [1],
    "ER_Status": [1],
    "PR_Status": [0],
    "HER2_Status": [0],
    "Ki67_Index": [15],
    "Menopause_Status": [1],
    "Family_History": [0],
    "BMI": [23.4],
    "Smoking": [0],
    "Alcohol": [0],
    "Comorbidity_Diabetes": [0],
    "Comorbidity_Hypertension": [1],
    "Chemo_Therapy": [1],
    "Radio_Therapy": [1],
    "Hormone_Therapy": [1],
    "Target_Therapy": [0],
    "Stage": [2],
    "Histological_Grade": [2],
    "Molecular_Subtype": [1],
    "Surgery_Type": [2],
    "Margin_Status": [0],
    "Lymph_Vascular_Invasion": [1],
    "Inflammatory_Response": [0],
    "Genetic_Test_Result": [0],
    "Followup_Months": [0],
    "Recurrence_Event": [0],
    "Blood_CA153": [25],
    "Blood_CEA": [3],
    "Blood_CA125": [20],
    "Blood_CA199": [15],
    "Blood_CA724": [5],
    "Blood_CA242": [8]
}
example_csv = pd.DataFrame(example_data)
buffer = io.BytesIO()
example_csv.to_csv(buffer, index=False)

st.download_button(label="📄 下载示例 CSV 模板 / Download Example CSV Template", data=buffer.getvalue(), file_name="example_patient_data.csv", mime="text/csv")

# ---------------------------
# 输入方式 / Input Method
# ---------------------------
input_method = st.radio(
    "选择输入方式 / Choose Input Method:",
    ("单例输入 / Single Input", "批量上传 CSV / Batch Upload CSV"),
    horizontal=True
)

# ---------------------------
# 单例输入 / Single Input
# ---------------------------
if input_method == "单例输入 / Single Input":
    st.subheader("👤 单例信息输入 / Single Patient Information")
    cols = st.columns(3)
    user_input = {}
    for i, col_name in enumerate(example_data.keys()):
        with cols[i % 3]:
            user_input[col_name] = st.number_input(col_name, value=float(example_data[col_name][0]))
    if st.button("预测复发风险 / Predict Recurrence Risk"):
        st.success("✅ 预测完成 / Prediction complete (示例代码占位)")

# ---------------------------
# 批量上传 / Batch Upload
# ---------------------------
else:
    st.subheader("📁 批量上传患者数据 / Batch Upload Patient Data")
    uploaded_file = st.file_uploader("上传 CSV 文件 / Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("数据预览 / Data Preview:")
        st.dataframe(df.head())
        if st.button("批量预测 / Run Batch Prediction"):
            st.success("✅ 批量预测完成 / Batch prediction complete (示例代码占位)")

# ---------------------------
# 风险图与说明 / Visualization & Disclaimer
# ---------------------------
st.subheader("📊 风险随时间曲线 / Risk Over Time Visualization")
x = np.linspace(0, 5, 100)
y = 1 - np.exp(-0.15 * x)
fig, ax = plt.subplots()
ax.plot(x, y, label="Cumulative Recurrence Risk")
ax.set_xlabel("时间 / Time (Years)")
ax.set_ylabel("累积复发概率 / Cumulative Recurrence Probability")
ax.legend()
st.pyplot(fig)

st.markdown("""
---
### ⚠️ 免责声明 Disclaimer
本网页应用仅用于科研和教学展示，不可作为临床诊断或治疗决策依据。  
_This web application is for research and educational purposes only and must not be used for clinical diagnosis or treatment decisions._
""")
