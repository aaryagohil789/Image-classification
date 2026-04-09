import streamlit as st
import requests
from PIL import Image, ImageDraw

HF_TOKEN = st.secrets("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-101-panoptic"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_image(image_bytes, content_type):
    response = requests.post(
        API_URL,
        headers={**headers, "Content-Type": content_type},
        data=image_bytes
    )
    return response.json()


st.title("Image Segmentation App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    content_type = uploaded_file.type.split(";")[0]

    with st.spinner("Segmenting image..."):
        output = query_image(img_bytes, content_type)

    if isinstance(output, list):
        draw = ImageDraw.Draw(image)

        st.subheader("Detected Regions:")
        for obj in output:
            label = obj.get("label", "unknown")
            score = obj.get("score", 0)
            box = obj.get("box", {})

            x_min = box.get("xmin", 0)
            y_min = box.get("ymin", 0)
            x_max = box.get("xmax", 0)
            y_max = box.get("ymax", 0)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min), f"{label} {score:.2f}", fill="red")

            st.write(f"- {label}: {score:.2f}")

        st.image(image, caption="Segmented Output", use_container_width=True)
    else:
        st.write("Error or unexpected response:")
        st.write(output)