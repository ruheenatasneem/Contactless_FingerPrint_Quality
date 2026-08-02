import streamlit as st
import cv2
import numpy as np
from PIL import Image

from quality_assessment import quality_gate

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Fingerprint Quality Assessment",
    page_icon="🖐",
    layout="wide"
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🖐 Contactless Fingerprint Quality Assessment")
st.write("Upload a contactless fingerprint image to evaluate its quality.")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Quality Thresholds")

blur_threshold = st.sidebar.slider(
    "Blur Threshold",
    min_value=5.0,
    max_value=50.0,
    value=10.0,
    step=1.0
)

dark_threshold = st.sidebar.slider(
    "Minimum Brightness",
    min_value=20,
    max_value=100,
    value=50
)

bright_threshold = st.sidebar.slider(
    "Maximum Brightness",
    min_value=150,
    max_value=255,
    value=210
)

glare_threshold = st.sidebar.slider(
    "Maximum Glare Ratio",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.write("Developed using")
st.sidebar.write("• Python")
st.sidebar.write("• OpenCV")
st.sidebar.write("• NumPy")
st.sidebar.write("• Streamlit")

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose Fingerprint Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# Main Section
# -------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image_np = np.array(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    result = quality_gate(image_bgr)

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # Left Column
    # ---------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width="stretch"
        )

    # ---------------------------------------------------
    # Right Column
    # ---------------------------------------------------

    with col2:

        st.subheader("Quality Assessment")

        score = result["composite_score"]

        st.metric(
            label="Composite Score",
            value=f"{score}/100"
        )

        if result["passed"]:

            st.success("✅ PASS")

        else:

            st.error("❌ FAIL")

        st.info(result["guidance"])

        st.progress(min(score / 100, 1.0))

    st.markdown("---")

    # ---------------------------------------------------
    # Metric Cards
    # ---------------------------------------------------

    st.subheader("Quality Metrics")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.write("### Blur")

        if result["blur"]["is_blurry"]:

            st.error("FAIL")

        else:

            st.success("PASS")

        st.write(
            f"Score : {result['blur']['blur_score']}"
        )

    with c2:

        st.write("### Brightness")

        if (
            result["brightness"]["too_dark"]
            or
            result["brightness"]["too_bright"]
        ):

            st.error("FAIL")

        else:

            st.success("PASS")

        st.write(
            f"Brightness : {result['brightness']['brightness']}"
        )

    with c3:

        st.write("### Glare")

        if result["glare"]["has_glare"]:

            st.error("FAIL")

        else:

            st.success("PASS")

        st.write(
            f"Ratio : {result['glare']['glare_fraction']}"
        )

    st.markdown("---")

    c4, c5 = st.columns(2)

    with c4:

        st.write("### ROI Completeness")

        if result["roi"]["roi_complete"]:

            st.success("PASS")

        else:

            st.error("FAIL")

        st.write(
            f"ROI : {result['roi']['roi_fraction']}"
        )

    with c5:

        st.write("### Ridge Clarity")

        if result["ridge"]["ridges_clear"]:

            st.success("PASS")

        else:

            st.error("FAIL")

        st.write(
            f"Score : {result['ridge']['ridge_score']}"
        )

    st.markdown("---")

    # ---------------------------------------------------
    # Summary Table
    # ---------------------------------------------------

    st.subheader("Summary")

    summary = {
        "Metric": [
            "Blur",
            "Brightness",
            "Glare",
            "ROI",
            "Ridge"
        ],

        "Status": [

            "FAIL" if result["blur"]["is_blurry"] else "PASS",

            "FAIL"
            if (
                result["brightness"]["too_dark"]
                or
                result["brightness"]["too_bright"]
            )
            else "PASS",

            "FAIL"
            if result["glare"]["has_glare"]
            else "PASS",

            "PASS"
            if result["roi"]["roi_complete"]
            else "FAIL",

            "PASS"
            if result["ridge"]["ridges_clear"]
            else "FAIL"
        ]
    }

    st.table(summary)

    st.markdown("---")

    st.subheader("Guidance")

    st.success(result["guidance"])

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

else:

    st.info("Please upload a fingerprint image.")

st.markdown("---")

st.caption(
    "Contactless Fingerprint Quality Assessment using OpenCV and Streamlit"
)