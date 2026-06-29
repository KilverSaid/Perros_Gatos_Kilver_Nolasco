import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Perros y Gatos",
    layout="centered"
)

st.title("Clasificador de Perros y Gatos")
st.write("Suba una imagen para identificar si corresponde a un perro o un gato.")

# Configuración
IMG_SIZE = (224, 224)

MODEL_DIR = Path("modelo_perros_gatos_mobilenet")

CLASS_PATH = MODEL_DIR / "class_names.json"

MODEL_PATHS = [
    MODEL_DIR / "perros_gatos_net.h5",
    MODEL_DIR / "perros_gatos_net.keras"
]

# Traducción de clases
LABELS_ES = {
    "Perros": "Perro",
    "Gatos": "Gato"
}


@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)

    st.error("No se encontró el modelo.")
    st.stop()


@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return ["Perros", "Gatos"]


def preparar_imagen(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    # Igual que en el entrenamiento
    img = np.array(img, dtype=np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def predecir(img):
    pred = modelo.predict(preparar_imagen(img), verbose=0)[0]

    top = np.argsort(pred)[::-1]

    resultados = []

    for i in top:
        resultados.append(
            (
                LABELS_ES.get(clases[i], clases[i]),
                float(pred[i] * 100)
            )
        )

    return resultados


# Cargar recursos
modelo = cargar_modelo()
clases = cargar_clases()

# Interfaz
archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen cargada",
        use_container_width=True
    )

    resultados = predecir(imagen)

    st.subheader("Resultado")

    st.success(
        f"Predicción: {resultados[0][0]} ({resultados[0][1]:.2f}%)"
    )

    st.write("Probabilidades:")

    for clase, prob in resultados:
        st.write(f"**{clase}:** {prob:.2f}%")

else:
    st.info("Cargue una imagen para comenzar.")
