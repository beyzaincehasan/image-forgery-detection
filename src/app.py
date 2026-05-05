import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input

XCEPTION_MODEL_PATH = "/home/beyza/image-forgery-detection/models/best_xception_model_q95.keras"
MOBILENET_MODEL_PATH = "/home/beyza/image-forgery-detection/models/mobilenet_ela_model.keras"
IMG_SIZE = (299, 299)

st.set_page_config(page_title="Image Forgery Detection", layout="wide")

@st.cache_resource
def load_models():
    xception_model = tf.keras.models.load_model(XCEPTION_MODEL_PATH)
    mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
    return xception_model, mobilenet_model

xception_model, mobilenet_model = load_models()

def convert_to_ela(image, quality=90):
    temp_path = "/tmp/temp_streamlit_ela.jpg"
    image = image.convert("RGB")
    image.save(temp_path, "JPEG", quality=quality)
    compressed = Image.open(temp_path)

    ela = ImageChops.difference(image, compressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela

def predict_image(image):
    ela_image = convert_to_ela(image)

    # Xception prediction
    x_img = ela_image.resize((299, 299))
    x_arr = np.array(x_img)
    x_arr = np.expand_dims(x_arr, axis=0)
    x_arr = preprocess_input(x_arr)
    xception_pred = xception_model.predict(x_arr)[0][0]

    # MobileNet prediction
    m_img = ela_image.resize((224, 224))
    m_arr = np.array(m_img)
    m_arr = np.expand_dims(m_arr, axis=0)
    m_arr = mobilenet_preprocess(m_arr)
    mobilenet_pred = mobilenet_model.predict(m_arr)[0][0]

    final_score = (xception_pred + mobilenet_pred) / 2

    if final_score >= 0.65:
        label = "Sahte / Manipüle Edilmiş Görüntü"
        confidence = final_score * 100

    elif final_score <= 0.54:
        label = "Gerçek Görüntü"
        confidence = (1 - final_score) * 100

    else:
        label = "Kararsız / Şüpheli Görüntü"
        confidence = abs(final_score - 0.54) * 100

    return label, confidence, ela_image, xception_pred, mobilenet_pred, final_score


def draw_keypoints(image, algorithm):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if algorithm == "SIFT":
        detector = cv2.SIFT_create()
    elif algorithm == "AKAZE":
        detector = cv2.AKAZE_create()
    elif algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=1000)
    else:
        return None, 0

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    result = cv2.drawKeypoints(
        img_bgr,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, len(keypoints)

st.title("Görüntü Sahteciliği Tespit Sistemi")
st.write("Bu sistem, ELA ön işleme ve Xception derin öğrenme modeli kullanarak görüntünün gerçek ya da sahte olduğunu tahmin eder.")

uploaded_file = st.file_uploader(
    "Bir görüntü yükleyin",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Yüklenen Görüntü")
    st.image(image, use_container_width=True)

    if st.button("Analiz Et"):
        label, confidence, ela_image, xception_pred, mobilenet_pred, final_score = predict_image(image)

        st.subheader("Model Tahmini")

        if "Sahte" in label:
            st.error(label)
        else:
            st.success(label)

        st.metric("Güven Skoru", f"%{confidence:.2f}")
        st.write(f"Xception skoru: {xception_pred:.4f}")
        st.write(f"MobileNetV2 skoru: {mobilenet_pred:.4f}")
        st.write(f"Ensemble final skoru: {final_score:.4f}")
        st.subheader("ELA Görüntüsü")
        st.image(ela_image, use_container_width=True)

        st.subheader("Klasik Özellik Çıkarım Algoritmaları")

        col1, col2, col3 = st.columns(3)

        with col1:
            sift_img, sift_count = draw_keypoints(image, "SIFT")
            st.write(f"SIFT Keypoint Sayısı: {sift_count}")
            st.image(sift_img, use_container_width=True)

        with col2:
            akaze_img, akaze_count = draw_keypoints(image, "AKAZE")
            st.write(f"AKAZE Keypoint Sayısı: {akaze_count}")
            st.image(akaze_img, use_container_width=True)

        with col3:
            orb_img, orb_count = draw_keypoints(image, "ORB")
            st.write(f"ORB Keypoint Sayısı: {orb_count}")
            st.image(orb_img, use_container_width=True)

        st.info(
            "SURF algoritması OpenCV hazır paketlerinde patent/nonfree kısıtı nedeniyle devre dışı olabilir. "
            "Bu nedenle arayüzde SIFT, AKAZE ve ORB algoritmaları aktif olarak kullanılmıştır."
        )
