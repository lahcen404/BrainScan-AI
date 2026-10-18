
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="BrainScan AI | Neural Imaging",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "brainscan_model.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# -------------------- DESIGN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070b14;
    --panel: #101725;
    --border: #273449;
    --cyan: #64f4df;
    --muted: #93a4bc;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #17313d 0%, transparent 35%),
        radial-gradient(ellipse at 0% 35%, #17152e 0%, transparent 30%),
        var(--bg);
    color: #edf5ff;
    font-family: 'Manrope', sans-serif;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    font-family: 'Manrope', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0a101c;
    border-right: 1px solid #202c3e;
}

[data-testid="stSidebar"] p {
    color: var(--muted);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.brand-icon {
    background: linear-gradient(145deg, #64f4df, #70bfff);
    color: #071018;
    border-radius: 14px;
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.brand-name {
    color: #f3f7ff;
    font-size: 19px;
    font-weight: 800;
}

.brand-sub {
    color: #7f93ad;
    font: 10px 'DM Mono', monospace;
    letter-spacing: 2px;
}

.eyebrow {
    color: var(--cyan);
    font: 11px 'DM Mono', monospace;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 36px;
    border-radius: 24px;
    border: 1px solid #2b4054;
    background: linear-gradient(115deg, #111d2b 0%, #101827 58%, #142b34 100%);
    margin-bottom: 28px;
}

.hero:after {
    content: "NEURAL / 01";
    position: absolute;
    right: 25px;
    top: 24px;
    color: #557085;
    font: 11px 'DM Mono', monospace;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: clamp(32px, 4vw, 54px);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -2px;
    margin: 15px 0 12px;
}

.hero h1 span {
    color: var(--cyan);
}

.hero p {
    max-width: 630px;
    color: #a7b7ca;
    font-size: 15px;
    line-height: 1.8;
}

.pill {
    display: inline-block;
    border: 1px solid #355467;
    color: #a5f9eb;
    background: #142b36;
    border-radius: 30px;
    padding: 7px 12px;
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1px;
}

.section-title {
    color: #edf5ff;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-sub {
    color: #8396ae;
    font-size: 12px;
    margin-bottom: 20px;
}

.panel {
    background: linear-gradient(145deg, #111a29, #0d1421);
    border: 1px solid #26364b;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-label {
    color: #8da3ba;
    font: 11px 'DM Mono', monospace;
    letter-spacing: 1.4px;
    margin-bottom: 14px;
}

.stat {
    background: #101a28;
    border: 1px solid #28394d;
    border-radius: 16px;
    padding: 17px;
    min-height: 95px;
}

.stat-label {
    color: #91a4ba;
    font-size: 11px;
}

.stat-value {
    color: var(--cyan);
    font-size: 22px;
    font-weight: 800;
    margin-top: 7px;
}

div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    background: linear-gradient(100deg, #64f4df, #8bdcfb);
    color: #07131d;
    font-weight: 800;
    min-height: 48px;
    transition: all .2s ease;
}

div.stButton > button:hover {
    box-shadow: 0 0 25px #64f4df38;
    transform: translateY(-1px);
    color: #07131d;
}

[data-testid="stFileUploader"] {
    background: #0c1522;
    border: 1px dashed #42637a;
    border-radius: 15px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #64f4df, #8bdcfb);
}

.result-class {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f0f7ff;
    margin: 7px 0;
}

.result-confidence {
    color: var(--cyan);
    font: 36px 'DM Mono', monospace;
    margin: 8px 0;
}

.mini-note {
    color: #91a4ba;
    font-size: 12px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #687c94;
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.2px;
    padding: 18px 0;
}

hr {
    border-color: #243246;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">⌘</div>
        <div>
            <div class="brand-name">BrainScan AI</div>
            <div class="brand-sub">NEURAL IMAGING LAB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Analysis Studio", "About the project"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="eyebrow">Model classes</div>', unsafe_allow_html=True)
    st.markdown("🔴 **Glioma**")
    st.markdown("🟠 **Meningioma**")
    st.markdown("🟢 **No tumor**")
    st.markdown("🔵 **Pituitary**")

    st.divider()
    st.markdown(
        '<div class="mini-note">CNN · 224 × 224 RGB<br>Experimental image classification</div>',
        unsafe_allow_html=True,
    )


# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
    <div class="pill">✦ &nbsp; AI-POWERED IMAGE ANALYSIS</div>
    <h1>See beyond the<br><span>image.</span></h1>
    <p>
        Explore brain MRI classification through a convolutional neural network.
        Upload a scan, run the model, and inspect its predicted class probabilities.
    </p>
    <div class="eyebrow">SCAN · ANALYZE · EXPLORE</div>
</div>
""", unsafe_allow_html=True)


if page == "About the project":
    st.markdown('<div class="section-title">About BrainScan AI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
        <div class="eyebrow">PROJECT OVERVIEW</div>
        <h3>A CNN image-classification project</h3>
        <p class="mini-note">
            BrainScan AI uses a trained convolutional neural network to classify
            brain MRI images into four dataset categories. The interface displays
            the model's predicted class and its output probabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="stat"><div class="stat-label">INPUT SIZE</div><div class="stat-value">224 × 224</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="stat"><div class="stat-label">MODEL TYPE</div><div class="stat-value">CNN</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="stat"><div class="stat-label">OUTPUT CLASSES</div><div class="stat-value">04</div></div>', unsafe_allow_html=True)

else:
    # -------------------- DASHBOARD --------------------
    st.markdown('<div class="section-title">Analysis studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload an image to begin a new model prediction.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-label">01 / IMAGE INPUT</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop an MRI image here",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Accepted formats: JPG, JPEG, PNG and BMP",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded MRI scan", use_container_width=True)
                st.caption(f"Image dimensions · {image.width} × {image.height}")
            except Exception:
                st.error("Unable to open this image. Please upload another file.")

        if image is None:
            st.markdown("""
            <div class="panel">
                <div style="font-size:36px">⌁</div>
                <h3>Ready for your scan</h3>
                <p class="mini-note">
                    Choose a brain MRI image using the uploader above.
                    The preview will appear here before analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-label">02 / MODEL OUTPUT</div>', unsafe_allow_html=True)

        if image is not None:
            if st.button("✦  Analyze MRI", use_container_width=True):
                try:
                    with st.spinner("Running neural analysis..."):
                        model = load_model()
                        image_array = np.array(image.resize(IMAGE_SIZE))
                        image_array = image_array.astype(np.float32) / 255.0
                        image_array = np.expand_dims(image_array, axis=0)

                        probabilities = model.predict(image_array, verbose=0)[0]
                        predicted_idx = int(np.argmax(probabilities))
                        predicted_class = CLASS_NAMES[predicted_idx]
                        confidence = float(probabilities[predicted_idx]) * 100

                        st.session_state["prediction"] = {
                            "class": predicted_class,
                            "confidence": confidence,
                            "probabilities": probabilities.tolist(),
                        }
                except Exception as error:
                    st.error(f"Analysis failed: {error}")

        result = st.session_state.get("prediction")

        if result:
            display_name = result["class"].replace("notumor", "No tumor").title()

            st.markdown(f"""
            <div class="panel">
                <div class="eyebrow">PREDICTION GENERATED</div>
                <div class="result-class">{display_name}</div>
                <div class="mini-note">Highest model output probability</div>
                <div class="result-confidence">{result["confidence"]:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="panel-label">CLASS PROBABILITIES</div>', unsafe_allow_html=True)

            for name, probability in zip(CLASS_NAMES, result["probabilities"]):
                display_name = name.replace("notumor", "No tumor").title()
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'gap:10px;margin-top:12px;margin-bottom:5px">'
                    f'<span>{display_name}</span>'
                    f'<span style="color:#64f4df;font-family:monospace">'
                    f'{probability * 100:.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(float(probability))

        else:
            st.markdown("""
            <div class="panel">
                <div class="eyebrow">AWAITING INPUT</div>
                <h3>Your results will appear here</h3>
                <p class="mini-note">
                    Upload an MRI image and select Analyze MRI to view
                    the predicted class and all four model output probabilities.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">IMPORTANT INFORMATION</div>
        <p class="mini-note">
            This is an educational AI project, not a clinically validated
            diagnostic system. Model probabilities are not guarantees of
            accuracy. Do not use this tool to diagnose or guide treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    BRAINSCAN AI &nbsp;·&nbsp; NEURAL IMAGING LAB &nbsp;·&nbsp; EDUCATIONAL PROJECT
</div>
""", unsafe_allow_html=True)
