import streamlit as st
import requests
from PIL import Image
import base64
import io
import random

HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-101-panoptic"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

COCO_PANOPTIC_LABELS = {
    0: "background",
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 13: "stop sign",
    14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    26: "backpack", 27: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis",
    36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",

    # stuff classes (important for segmentation)
    92: "banner", 93: "blanket", 94: "bridge", 95: "cardboard",
    96: "counter", 97: "curtain", 98: "door-stuff",
    99: "floor-wood", 100: "flower", 101: "fruit",
    102: "gravel", 103: "house", 104: "light",
    105: "mirror-stuff", 106: "net", 107: "pillow",
    108: "platform", 109: "playingfield", 110: "railroad",
    111: "river", 112: "road", 113: "roof",
    114: "sand", 115: "sea", 116: "shelf",
    117: "snow", 118: "stairs", 119: "tent",
    120: "towel", 121: "wall-brick", 122: "wall-stone",
    123: "wall-tile", 124: "wall-wood", 125: "water",
    126: "window-blind", 127: "window",
    128: "tree", 129: "fence", 130: "ceiling",
    131: "sky", 132: "cabinet", 133: "table",
}

def query_image(image_bytes, content_type):
    response = requests.post(
        API_URL,
        headers={**headers, "Content-Type": content_type},
        data=image_bytes
    )
    return response.json()


def decode_mask(mask_base64):
    mask_bytes = base64.b64decode(mask_base64)
    return Image.open(io.BytesIO(mask_bytes)).convert("L")


def random_color():
    return [random.randint(0, 255) for _ in range(3)]


st.set_page_config(layout="wide")
st.title("Panoptic Image Segmentation")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    content_type = uploaded_file.type.split(";")[0]

    with st.spinner("Segmenting..."):
        output = query_image(img_bytes, content_type)

    if isinstance(output, list):
        overlay = image.copy()

        for obj in output:
            mask = decode_mask(obj["mask"])
            color = random_color()

            mask = mask.resize(overlay.size)

            # apply color overlay
            overlay_pixels = overlay.load()
            mask_pixels = mask.load()

            for i in range(overlay.size[0]):
                for j in range(overlay.size[1]):
                    if mask_pixels[i, j] > 0:
                        r, g, b = overlay_pixels[i, j]
                        overlay_pixels[i, j] = (
                            int(0.6 * r + 0.4 * color[0]),
                            int(0.6 * g + 0.4 * color[1]),
                            int(0.6 * b + 0.4 * color[2]),
                        )

        with col2:
            st.subheader("Segmented Output")
            st.image(overlay, use_container_width=True)

            st.subheader("Detected Segments")
            for obj in output:
                raw_label = obj.get("label", "")

                # Case 1: Label_yyy format
                if isinstance(raw_label, str) and raw_label.startswith("LABEL_"):
                    try:
                        label_id = int(raw_label.split("_")[1])
                    except:
                        label_id = None
                else:
                    label_id = None

                # Map if possible, else fallback
                if label_id is not None:
                    label_name = COCO_PANOPTIC_LABELS.get(label_id, raw_label)
                else:
                    label_name = raw_label  # already readable or unknown

                st.write(f"{label_name} → {obj.get('score', 0):.2f}")

    else:
        st.error("API Error")
        st.write(output)