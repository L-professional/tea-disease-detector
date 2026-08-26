import streamlit as st
import cv2
import os
import gdown
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Chẩn Đoán Bệnh Lá Chè", page_icon="🌱")

@st.cache_resource
def load_model():
    model_file = "best.pt"
    if not os.path.exists(model_file):
        # ⚠️ ĐIỀN FILE ID TỪ GOOGLE DRIVE CỦA BẠN VÀO GIỮA 2 DẤU NGOẶC KÉP
        file_id = "19oOI04TLvndit8bhlQcWHnLcYukkAD3P"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_file, quiet=False)
    return YOLO(model_file)

model = load_model()

DISEASE_TRANSLATE = {
    'algal_spot': 'Bệnh đốm tảo',
    'brown_blight': 'Bệnh thối nâu (Cháy lá nâu)',
    'gray_blight': 'Bệnh chấm xám (Thán thư)',
    'healthy': 'Lá khỏe mạnh',
    'helopeltis': 'Bọ xít muỗi gây hại',
    'red_spot': 'Bệnh đốm đỏ'
}

st.title("🌱 Hệ Thống Chẩn Đoán Bệnh Lá Chè")
conf_threshold = st.slider("Ngưỡng độ tin cậy", 0.1, 0.9, 0.25, 0.05)

# Chia giao diện thành 2 tab: Chụp ảnh & Tải tệp
tab1, tab2 = st.tabs(["📸 Chụp ảnh trực tiếp", "📁 Tải ảnh từ thiết bị"])

img_file = None

with tab1:
    camera_photo = st.camera_input("Chụp ảnh lá chè từ ống kính camera")
    if camera_photo:
        img_file = camera_photo

with tab2:
    upload_photo = st.file_uploader("Chọn tệp ảnh lá chè...", type=["jpg", "jpeg", "png"])
    if upload_photo:
        img_file = upload_photo

if img_file is not None:
    image = Image.open(img_file)
    if st.button("🔍 Tiến Hành Chẩn Đoán", type="primary"):
        results = model.predict(source=image, conf=conf_threshold)
        r = results[0]
        annotated_img = cv2.cvtColor(r.plot(line_width=4), cv2.COLOR_BGR2RGB)
        st.image(annotated_img, caption="Kết quả khoanh vùng vết bệnh", use_container_width=True)
        
        if len(r.boxes) == 0:
            st.info("⚠️ Không phát hiện vết bệnh (Lá khỏe mạnh hoặc thử hạ thấp ngưỡng độ tin cậy).")
        else:
            for i, box in enumerate(r.boxes, 1):
                cls_id = int(box.cls[0])
                cls_en = model.names[cls_id]
                cls_vn = DISEASE_TRANSLATE.get(cls_en, cls_en)
                conf = float(box.conf[0]) * 100
                st.success(f"**Vết bệnh #{i}:** {cls_vn.upper()} — Độ tin cậy: **{conf:.1f}%**").1, 0.9, 0.25, 0.05)
img_file = st.file_uploader("Tải ảnh hoặc chụp ảnh lá chè...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    image = Image.open(img_file)
    if st.button("🔍 Tiến Hành Chẩn Đoán", type="primary"):
        results = model.predict(source=image, conf=conf_threshold)
        r = results[0]
        annotated_img = cv2.cvtColor(r.plot(line_width=4), cv2.COLOR_BGR2RGB)
        st.image(annotated_img, caption="Kết quả khoanh vùng vết bệnh", use_container_width=True)
        
        if len(r.boxes) == 0:
            st.info("⚠️ Không phát hiện vết bệnh (Lá khỏe mạnh hoặc thử hạ thấp ngưỡng độ tin cậy).")
        else:
            for i, box in enumerate(r.boxes, 1):
                cls_id = int(box.cls[0])
                cls_en = model.names[cls_id]
                cls_vn = DISEASE_TRANSLATE.get(cls_en, cls_en)
                conf = float(box.conf[0]) * 100
                st.success(f"**Vết bệnh #{i}:** {cls_vn.upper()} — Độ tin cậy: **{conf:.1f}%**")
