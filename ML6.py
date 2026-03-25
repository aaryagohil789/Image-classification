import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/deit-base-distilled-patch16-224"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_image(image_bytes, content_type="image/jpeg"):
    response = requests.post(
        API_URL,
        headers={"Content-Type": content_type, **headers},
        data=image_bytes
    )
    return response.json()

st.title("Image Classification")
st.write("Upload an image to predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    content_type = "image/png" if uploaded_file.type == "image/png" else "image/jpeg"

    with st.spinner("Classifying..."):
        output = query_image(img_bytes, content_type)

    st.subheader("Predictions:")

    if isinstance(output, list):
        for item in output:
            label = item["label"]
            score = item["score"]
            st.write(f"**{label}** — confidence: {score:.2%}")
    else:
        st.write("Unexpected output or API error:")
        st.json(output)