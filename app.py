import streamlit as st
from PIL import Image
import numpy as np
import time
from detector import detect_damage

# Page Configuration
st.set_page_config(
    page_title="Road Damage Detection AI",
    page_icon="🚧",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #94a3b8;
    margin-bottom: 35px;
}

.upload-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #334155;
}

.result-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #334155;
}

.metric-card {
    background: #1e293b;
    padding: 15px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #334155;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding-top: 40px;
    font-size: 15px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="title">🚧 Road Damage Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Road Damage Analysis using YOLOv8</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:

    st.markdown("## 📌 About Project")

    st.write("""
    This AI system detects road damages using Deep Learning and YOLOv8.

    ### Features
    - Upload road images
    - Capture image using camera
    - Real-time damage detection
    - Severity analysis
    - Smart AI inference
    """)

    st.success("🚀 Model: YOLOv8")
    st.info("💻 Framework: Streamlit")

    st.markdown("---")

    st.markdown("## 👨‍💻 Team Members")

    st.markdown("""
    ### 1. Ananta Narayan Sethy
    🔗 GitHub:  
    https://github.com/DevAnantaNsethy

    💼 LinkedIn:  
    https://www.linkedin.com/in/ananta-narayan-sethy-46403a24b/

    ---

    ### 2. Biplab Nayak
    💼 LinkedIn:  
    https://www.linkedin.com/in/biplab-nayak-a21525378/

    ---

    ### 3. Sourav Choudhuri
    💼 LinkedIn:  
    https://www.linkedin.com/in/sourav-choudhuri-52bb48299/

    ---

    ### 4. Lokanath Pahan
    💼 LinkedIn:  
    https://www.linkedin.com/in/lokanath-pahan-91542929b/
    """)

# Main UI
st.markdown("---")

# Input Selection
input_option = st.radio(
    "📥 Choose Input Method",
    ["📤 Upload Image", "📷 Use Camera"],
    horizontal=True
)

uploaded_image = None

# Upload Option
if input_option == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

# Camera Option
else:

    camera_photo = st.camera_input("Capture Road Image")

    if camera_photo:
        uploaded_image = Image.open(camera_photo).convert("RGB")

# Image Processing
if uploaded_image:

    st.markdown("## 📷 Selected Image")

    # Fake loading animation for better UX
    progress_bar = st.progress(0)

    for percent in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent + 1)

    st.image(
        uploaded_image,
        caption="Input Image",
        use_container_width=True
    )

    img_array = np.array(uploaded_image)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Road Condition", use_container_width=True):

        with st.spinner("🧠 AI Model is analyzing the road image..."):

            # Extra loading animation
            loading_text = st.empty()

            for i in range(3):
                loading_text.markdown(
                    f"### 🔍 Detecting damages{'.' * (i+1)}"
                )
                time.sleep(0.5)

            result_img, results = detect_damage(img_array.copy())

            loading_text.empty()

        st.success("✅ Analysis Completed Successfully")

        st.markdown("---")

        # Result Image
        st.markdown("## 🧠 Detection Result")

        st.image(
            result_img,
            use_container_width=True
        )

        st.markdown("---")

        # Detection Summary
        st.markdown("## 📊 Detection Summary")

        damage_count = 0
        confidences = []
if results is not None:
    for data in results.boxes.data.tolist():

            conf = float(data[4])

            if conf >= 0.3:
                damage_count += 1
                confidences.append(conf)

        # Metrics
        col1, col2, col3 = st.columns(3)

        if damage_count == 0:

            with col1:
                st.metric("Road Status", "GOOD")

            with col2:
                st.metric("Damages", "0")

            with col3:
                st.metric("Severity", "LOW")

            st.success("🛣️ No road damage detected.")

        else:

            avg_conf = sum(confidences) / len(confidences)

            if damage_count == 1:
                severity = "LOW"

            elif damage_count <= 3:
                severity = "MEDIUM"

            else:
                severity = "HIGH"

            with col1:
                st.metric("Road Status", "DAMAGED")

            with col2:
                st.metric("Damages", damage_count)

            with col3:
                st.metric("Confidence", f"{avg_conf:.2f}")

            if severity == "LOW":
                st.info("🟢 Severity Level: LOW")

            elif severity == "MEDIUM":
                st.warning("🟡 Severity Level: MEDIUM")

            else:
                st.error("🔴 Severity Level: HIGH")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Team Road Damage Detection 🚀
    </div>
    """,
    unsafe_allow_html=True
)
