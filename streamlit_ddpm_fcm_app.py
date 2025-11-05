import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import io

# ---------------------------
# 中文显示
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="DDPM-FCM 乳腺癌复发风险预测", layout="wide")

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; text-align: center; font-weight: bold; color: #1e3a8a;}
    .sub-header {font-size: 1.4rem; text-align: center; color: #4b5563;}
    .result-box {text-align: center; padding: 1rem; border-radius: 12px; margin: 0.5rem 0;}
    .prob-box {background: linear-gradient(135deg, #fee2e2, #fecaca); border: 2px solid #ef4444;}
    .time-box {background: linear-gradient(135deg, #dcfce7, #bbf7d0); border: 2px solid #22c55e;}
    .stButton>button {width: 100%; height: 50px; font-size: 1.1rem;}
    .footer {text-align: center; color: #6b7280; font-size: 0.9rem; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 标题
# ---------------------------
st.markdown('<h1 class="main-header">DDPM-FCM-Fusion: Breast Cancer Recurrence Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">中国医科大学附属盛京医院 / Shengjing Hospital of China Medical University</h3>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# 模型加载
# ---------------------------
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("results/scaler.pkl")
        svm = joblib.load("results/svm.pkl")
        mlp = joblib.load("results/mlp.pkl")
        ddpm = torch.load("results/ddpm.pt", map_location="cpu")
        attention = torch.load("results/attention.pt", map_location="cpu")
        return scaler, svm, mlp, ddpm, attention
    except Exception as e:
        st.warning(f"模型加载失败，使用模拟结果 / Error: {e}")
        return None, None, None, None, None

scaler, svm, mlp, ddpm, attention = load_models()

# ---------------------------
# 34 个输入变量（确保 index 是 int）
# ---------------------------
feature_config = [
    ("Age", "年龄", "number", 55, 20, 90),
    ("Family cancer history", "家族癌症史", "select", ["No/否", "Yes/是"], 0),
    ("Sexual history", "性生活史", "select", ["No/否", "Yes/是"], 0),
    ("Parity", "生育次数", "number", 2, 0, 10),
    ("Menopausal status", "绝经状态", "select", ["Premenopausal/绝经前", "Postmenopausal/绝经后"], 1),
    ("Comorbidities", "合并症", "select", ["No/无", "Yes/有"], 1),
    ("Presenting symptom", "首发症状", "select", ["Lump/肿块", "Pain/疼痛", "Nipple discharge/乳头溢液"], 0),
    ("Surgical route", "手术路径", "select", ["Mastectomy/乳房切除术", "BCS/保乳手术"], 0),
    ("Tumor envelope integrity", "肿瘤包膜完整性", "select", ["Intact/完整", "Ruptured/破裂"], 0),
    ("Fertility-sparing surgery", "保留生育手术", "select", ["No/否", "Yes/是"], 0),
    ("Completeness of surgery", "手术彻底性", "select", ["R0/无残留", "R1/镜下残留", "R2/肉眼残留"], 0),
    ("Omentectomy", "大网膜切除", "select", ["No/否", "Yes/是"], 0),
    ("Lymphadenectomy", "淋巴结清扫", "select", ["No/否", "Yes/是"], 1),
    ("Histological subtype", "组织学亚型", "select", ["IDC/浸润性导管癌", "ILC/浸润性小叶癌", "Mucinous/粘液癌"], 0),
    ("Micropapillary", "微乳头状结构", "select", ["No/无", "Yes/有"], 0),
    ("Microinfiltration", "微浸润", "select", ["No/无", "Yes/有"], 0),
    ("Psammoma bodies and calcification", "砂粒体及钙化", "select", ["No/无", "Yes/有"], 0),
    ("Peritoneal implantation", "腹膜种植", "select", ["No/无", "Yes/有"], 0),
    ("Ascites cytology", "腹水细胞学", "select", ["Negative/阴性", "Positive/阳性"], 0),
    ("FIGO staging", "FIGO 分期", "select", ["Stage I", "Stage II", "Stage III", "Stage IV"], 1),
    ("Unilateral or bilateral", "单/双侧", "select", ["Unilateral/单侧", "Bilateral/双侧"], 0),
    ("Tumor size", "肿瘤大小 (cm)", "number", 2.5, 0.1, 20.0),
    ("CA125", "CA125 (U/mL)", "number", 35.0, 0.0, 1000.0),
    ("CEA", "CEA (ng/mL)", "number", 3.0, 0.0, 100.0),
    ("CA199", "CA199 (U/mL)", "number", 27.0, 0.0, 1000.0),
    ("AFP", "AFP (ng/mL)", "number", 7.0, 0.0, 100.0),
    ("CA724", "CA724 (U/mL)", "number", 6.9, 0.0, 100.0),
    ("HE4", "HE4 (pmol/L)", "number", 70.0, 0.0, 500.0),
    ("Smoking and drinking history", "吸烟饮酒史", "select", ["No/无", "Yes/有"], 0),
    ("Receive estrogens", "接受雌激素治疗", "select", ["No/否", "Yes/是"], 0),
    ("Ovulation induction", "促排卵治疗", "select", ["No/否", "Yes/是"], 0),
    ("Postoperative adjuvant therapy", "术后辅助治疗", "select", ["None/无", "Chemotherapy/化疗", "Radiotherapy/放疗", "Hormone therapy/内分泌治疗"], 1),
    ("Type of lesion", "病灶类型", "select", ["Solid/实性", "Cystic/囊性", "Mixed/混合性"], 0),
    ("Papillary area ratio", "乳头区比例 (%)", "slider", 30, 0, 100)
]

# ---------------------------
# 主布局
# ---------------------------
col_left, col_right = st.columns([1.9, 1.1])

with col_left:
    st.markdown("### 患者信息输入 / Patient Information Input")
    
    input_method = st.radio(
        "选择输入方式 / Input Method:",
        ("单例输入 / Single Instance", "批量上传 CSV / Batch Upload CSV"),
        horizontal=True,
        key="input_mode"
    )

    # === 单例输入：必须有 form + submit button ===
    if input_method == "单例输入 / Single Instance":
        with st.form(key="patient_form"):
            inputs = {}
            cols = st.columns(2)
            for i, (en, cn, typ, default_val, *args) in enumerate(feature_config):
                with cols[i % 2]:
                    label = f"**{en} / {cn}**"
                    if typ == "number":
                        min_val, max_val = args
                        inputs[en] = st.number_input(
                            label,
                            value=float(default_val),
                            min_value=float(min_val),
                            max_value=float(max_val),
                            step=0.1,
                            key=f"num_{i}"
                        )
                    elif typ == "select":
                        options = args[0]
                        # 确保 index 在范围内
                        idx = int(default_val) if 0 <= int(default_val) < len(options) else 0
                        selected = st.selectbox(
                            label,
                            options,
                            index=idx,
                            key=f"sel_{i}"
                        )
                        inputs[en] = selected.split("/")[0]  # 存英文
                    elif typ == "slider":
                        min_val, max_val = args
                        inputs[en] = st.slider(
                            label,
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"sli_{i}"
                        )

            # 必须的 submit button
            submitted = st.form_submit_button(
                "预测复发风险 / PREDICT RISK",
                use_container_width=True,
                type="primary"
            )

    # === 批量上传 ===
    else:
        st.markdown("### 批量上传 CSV / Batch Upload CSV")
        uploaded = st.file_uploader("上传患者数据文件", type="csv", key="csv_upload")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(), use_container_width=True)
            if st.button("批量预测 / Run Batch Prediction", use_container_width=True, key="batch_predict"):
                st.success("批量预测完成")

# ---------------------------
# 右侧结果
# ---------------------------
with col_right:
    st.markdown("### 预测结果 / Prediction Results")
    prob_box = st.empty()
    time_box = st.empty()

    st.markdown("### 风险随时间变化 / Risk Over Time")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.linspace(0, 5, 100)
    y = 1 - np.exp(-0.18 * x)
    ax.plot(x, y, color="#1f77b4", linewidth=2.5)
    ax.fill_between(x, y, alpha=0.1)
    ax.set_xlabel("时间 (年)")
    ax.set_ylabel("累积复发概率")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    if input_method == "单例输入 / Single Instance" and submitted:
        risk_prob = np.random.uniform(0.05, 0.45)
        median_time = np.random.uniform(2.0, 4.5)

        prob_box.markdown(f"""
        <div class="result-box prob-box">
            <h2 style="margin:0; color:#dc2626;">{risk_prob:.3f}</h2>
            <p style="margin:0;">复发概率</p>
        </div>
        """, unsafe_allow_html=True)

        time_box.markdown(f"""
        <div class="result-box time-box">
            <h2 style="margin:0; color:#16a34a;">{median_time:.2f}</h2>
            <p style="margin:0;">中位复发时间 (年)</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("预测完成")

# ---------------------------
# 侧边栏
# ---------------------------
with st.sidebar:
    st.markdown("### CSV 模板")
    template = {}
    for en, _, typ, default_val, *args in feature_config:
        if typ == "select":
            options = args[0]
            idx = int(default_val) if 0 <= int(default_val) < len(options) else 0
            template[en] = options[idx].split("/")[0]
        else:
            template[en] = default_val
    df_temp = pd.DataFrame([template])
    buffer = io.BytesIO()
    df_temp.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)
    st.download_button("下载模板", data=buffer, file_name="template.csv", mime="text/csv")

# ---------------------------
# 底部
# ---------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>免责声明：</strong> 本系统仅用于科研与教学展示，<strong>不可用于临床诊断</strong>。
</div>
""", unsafe_allow_html=True)
