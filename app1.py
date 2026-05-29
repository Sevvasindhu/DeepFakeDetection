import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from collections import deque
from mtcnn import MTCNN
from retinaface import RetinaFace

# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 224
THRESHOLD = 0.7
SMOOTHING_FRAMES = 10
LAST_CONV_LAYER = "block14_sepconv2_act"

st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🧠",
    layout="centered"
)

# ======================================================
# SESSION STATE
# ======================================================
if "camera_state" not in st.session_state:
    st.session_state.camera_state = "STOPPED"

# ======================================================
# HEADER UI
# ======================================================
st.markdown(
    """
    <h1 style="text-align:center;">🧠 DeepFake Detection System</h1>
    <p style="text-align:center; color:gray;">
    Image Upload & Real-Time Detection with Grad-CAM
    </p>
    """,
    unsafe_allow_html=True
)

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn.keras")
    xception = tf.keras.models.load_model("xception.keras")
    return cnn, xception

cnn_model, xception_model = load_models()

# ======================================================
# FACE DETECTOR
# ======================================================
detector_type = st.selectbox(
    "👤 Select Face Detector",
    ["MTCNN",]
)

@st.cache_resource
def load_detector(name):
    return MTCNN() if "MTCNN" in name else RetinaFace

detector = load_detector(detector_type)

# ======================================================
# PREPROCESS
# ======================================================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ======================================================
# ENSEMBLE PREDICTION
# ======================================================
def ensemble_predict(img):
    inp = preprocess(img)
    p1 = cnn_model.predict(inp, verbose=0)[0][0]
    p2 = xception_model.predict(inp, verbose=0)[0][0]
    final = (p1 + p2) / 2

    label = "FAKE" if final > THRESHOLD else "REAL"
    confidence = final * 100 if label == "FAKE" else (100 - final * 100)
    confidence = min(confidence, 99.0)

    return label, confidence

# ======================================================
# GRAD-CAM
# ======================================================
def gradcam(image):
    img_tensor = preprocess(image)

    grad_model = tf.keras.models.Model(
        [xception_model.inputs],
        [
            xception_model.get_layer(LAST_CONV_LAYER).output,
            xception_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# ======================================================
# MODE SELECTION
# ======================================================
mode = st.radio(
    "📌 Select Mode",
    ["🖼️ Upload Image", "📷 Real-Time Webcam"],
    horizontal=True
)

st.divider()

# ======================================================
# IMAGE UPLOAD MODE
# ======================================================
if "Upload" in mode:
    uploaded = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = np.array(Image.open(uploaded).convert("RGB"))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect DeepFake"):
            if "MTCNN" in detector_type:
                faces = detector.detect_faces(img)
                if faces:
                    x, y, w, h = faces[0]["box"]
                else:
                    faces = []
            else:
                faces = RetinaFace.detect_faces(img)

                if isinstance(faces, dict):
                    box = list(faces.values())[0]["facial_area"]
                    x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                else:
                    faces = []

            if len(faces) == 0:
                st.warning("No face detected")
            else:
                face = img[y:y+h, x:x+w]
                label, confidence = ensemble_predict(face)
                cam = gradcam(face)
                img[y:y+h, x:x+w] = cam

                st.image(img, caption="Grad-CAM Visualization", use_column_width=True)

                st.subheader("📊 Prediction Result")
                st.write(f"**Result:** {label}")
                st.progress(int(confidence))
                st.write(f"**Confidence:** {confidence:.2f}%")

# ======================================================
# REAL-TIME MODE
# ======================================================
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start"):
            st.session_state.camera_state = "RUNNING"
    with col2:
        if st.button("⏸ Pause"):
            st.session_state.camera_state = "PAUSED"
    with col3:
        if st.button("⏹ Stop"):
            st.session_state.camera_state = "STOPPED"

    frame_box = st.empty()
    result_box = st.empty()
    conf_box = st.empty()

    pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

    if st.session_state.camera_state != "STOPPED":
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and st.session_state.camera_state != "STOPPED":

            if st.session_state.camera_state == "PAUSED":
                continue

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if "MTCNN" in detector_type:
                faces = detector.detect_faces(rgb)
                if faces:
                    x, y, w, h = faces[0]["box"]
                else:
                    faces = []
            else:
                faces = RetinaFace.detect_faces(rgb)
                if isinstance(faces, dict):
                    box = list(faces.values())[0]["facial_area"]
                    x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                else:
                    faces = []

            if len(faces) > 0:
                face = rgb[y:y+h, x:x+w]
                label, confidence = ensemble_predict(face)
                pred_buffer.append(confidence)
                smooth_conf = sum(pred_buffer) / len(pred_buffer)

                cam = gradcam(face)
                rgb[y:y+h, x:x+w] = cam

                result_box.markdown(f"### Result: **{label}**")
                conf_box.progress(int(smooth_conf))
                conf_box.markdown(f"Confidence: **{smooth_conf:.2f}%**")
            else:
                result_box.warning("No face detected")

            frame_box.image(rgb)

        cap.release()
