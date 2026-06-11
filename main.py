# streamlit run <filename>
# example:
# streamlit run src/Apr1_CatVSDog.py


# To get this file on streamlit online, do the following:
# 1. Make sure the labelling function is the same
# 2. Make sure the file path is correct for the mode
# Reminder: do not store files in .venv
# just make a models directory


import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import io

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

def cat_or_dog(file_name):
    if file_name[0].isupper():
        return "CAT"
    else:
        return "DOG"

@st.cache_resource
def get_model():
    return load_learner("models/cat_vs_dog_model_fastai_2_8_4.pkl")

learn = get_model()

st.title("Cat vs Dog Classifier")
st.write("Upload an image and I’ll predict whether it’s a CAT or DOG.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    fastai_img = PILImage.create(pil_img)
    pred_class, pred_idx, probs = learn.predict(fastai_img)

    conf = float(probs[int(pred_idx)]) * 100.0
    st.subheader("Prediction")
    st.write(f"**{pred_class}**")
    st.write(f"Confidence: **{conf:.2f}%**")