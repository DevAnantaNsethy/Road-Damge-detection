import streamlit as st
from PIL import Image
import numpy as np
from detector import detect_damage, model

# Page config
st.set_page_config(page_title="Road Damage Detection", layout="centered")

# Custom CSS (🎨 Styling)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title"> Road Damage Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered detection using YOLOv8</p>', unsafe_allow_html=True)

st.write("---")

# Upload
uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",use_container_width=True)

    img_array = np.array(image)

    if st.button("Analyze Road Condition"):
        with st.spinner("Analyzing image..."):
            results = model(img_array)[0]
            result_img = detect_damage(img_array.copy())

        st.success("Analysis Complete ✅")

        st.image(result_img, caption="🧠 Detection Result", use_column_width=True)

        st.write("---")
        st.subheader(" Detection Summary")

        damage_count = 0
        confidences = []

        for data in results.boxes.data.tolist():
            conf = float(data[4])
            if conf >= 0.3:
                damage_count += 1
                confidences.append(conf)

        # Smart summary
        if damage_count == 0:
            st.success(" Road Condition: GOOD (No damage detected)")
        else:
            avg_conf = sum(confidences) / len(confidences)

            st.error(f" Road Condition: DAMAGED")
            st.write(f" Total Damages Detected: {damage_count}")
            st.write(f" Average Confidence: {avg_conf:.2f}")

            # Severity logic
            if damage_count == 1:
                st.info("🟢 Severity: LOW")
            elif damage_count <= 3:
                st.warning("🟡 Severity: MEDIUM")
            else:
                st.error("🔴 Severity: HIGH")

        st.write("---")