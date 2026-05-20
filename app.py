import streamlit as st
from PIL import Image
import numpy as np
from detector import detect_damage, model

# Page Configuration
st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown(
    '<p class="title">🚧 Road Damage Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-powered road damage analysis using YOLOv8</p>',
    unsafe_allow_html=True
)

st.write("---")

# Sidebar
with st.sidebar:

    st.header("📌 About Project")

    st.write("""
    Automated Road Damage Detection System using Deep Learning and YOLOv8.

    ### Features
    - Detects road damages
    - Severity analysis
    - Real-time AI inference
    - AI-powered road condition monitoring
    """)

    st.success("Model: YOLOv8")
    st.info("Framework: Streamlit")

    st.write("---")

    st.header("👨‍💻 Team Members")

    st.markdown("""
    ### 1. Ananta Narayan Sethy
    - GitHub: https://github.com/
    - LinkedIn: https://linkedin.com/

    ### 2. Biplab Nayak
    - GitHub: https://github.com/
    - LinkedIn: https://linkedin.com/

    ### 3. Sourav Choudhuri
    - GitHub: https://github.com/
    - LinkedIn: https://linkedin.com/

    ### 4. Lokanath Pahan
    - GitHub: https://github.com/
    - LinkedIn: https://linkedin.com/
    """)

# File Upload
uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "png", "jpeg"]
)

# Image Processing
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="📷 Uploaded Image",
        use_container_width=True
    )

    img_array = np.array(image)

    # Analyze Button
    if st.button("🔍 Analyze Road Condition"):

        with st.spinner("Analyzing image..."):

            results = model(img_array)[0]
            result_img = detect_damage(img_array.copy())

        st.success("✅ Analysis Complete")

        # Detection Result
        st.image(
            result_img,
            caption="🧠 Detection Result",
            use_container_width=True
        )

        st.write("---")

        st.subheader("📊 Detection Summary")

        damage_count = 0
        confidences = []

        # Detection Analysis
        for data in results.boxes.data.tolist():

            conf = float(data[4])

            if conf >= 0.3:
                damage_count += 1
                confidences.append(conf)

        # Smart Summary
        if damage_count == 0:

            st.success("🛣️ Road Condition: GOOD (No damage detected)")

        else:

            avg_conf = sum(confidences) / len(confidences)

            st.error("⚠️ Road Condition: DAMAGED")

            st.write(f"### Total Damages Detected: {damage_count}")
            st.write(f"### Average Confidence: {avg_conf:.2f}")

            # Severity Logic
            if damage_count == 1:

                st.info("🟢 Severity Level: LOW")

            elif damage_count <= 3:

                st.warning("🟡 Severity Level: MEDIUM")

            else:

                st.error("🔴 Severity Level: HIGH")

        st.write("---")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Team Road Damage Detection 🚀
    </div>
    """,
    unsafe_allow_html=True
)