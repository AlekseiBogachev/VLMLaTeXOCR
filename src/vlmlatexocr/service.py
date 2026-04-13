# poetry run streamlit run src/vlmlatexocr/cli.py run-service

from pathlib import Path

import streamlit as st
from PIL import Image

from vlmlatexocr.model import run_predict


def start_sevice():
    """Start a Streamlit service for LaTeX OCR."""
    st.set_page_config(page_title="VLMLaTeXOCR", layout="wide")
    st.title("Распознавание LaTeX формул")

    model_config = Path("./configs/model.json")

    uploaded_file = st.file_uploader(
        "Загрузите изображение с формулой", type=["jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", width=400)

        with st.spinner("Распознавание..."):
            prediction = run_predict(image, str(model_config))

        st.success("Распознавание завершено!")
        st.write("Предсказанный LaTeX:")
        st.code(prediction, language="latex")
