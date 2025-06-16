# --- LIBRERÍAS BASE ---
import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from openai import OpenAI

# --- VIDEO Y EMOCIONES ---
import cv2
from deepface import DeepFace
from scipy.signal import butter, lfilter

# --- LLM (NUEVA VERSIÓN OPENAI >= 1.0.0) ---
from openai import OpenAI

# --- INICIO APP ---
st.title("🔬 MVP de Neuromarketing - Análisis Facial y BPM por segundo")

# Subir video
video_file = st.file_uploader("📤 Sube tu video interactuando con el producto (.mp4)", type=["mp4"])
if not video_file:
    st.warning("👆 Por favor, sube un video para comenzar el análisis.")
    st.stop()

# Mostrar video
st.video(video_file)

# Guardar temporalmente
temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
temp_video.write(video_file.read())
video_path = temp_video.name

# Leer video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
st.write(f"🎥 FPS: {fps:.1f}, Duración: {duration:.1f}s, Frames: {total_frames}")

# Análisis por segundo
frame_emociones = []
bpm_signal = []

st.write("⏳ Analizando emociones y BPM...")

for segundo in range(int(duration)):
    cap.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        # --- Emoción facial ---
        temp_frame_path = os.path.join(tempfile.gettempdir(), f"frame_{segundo}.jpg")
        cv2.imwrite(temp_frame_path, frame)
        emo_result = DeepFace.analyze(temp_frame_path, actions=["emotion"], enforce_detection=False)
        emociones_completas = emo_result[0]["emotion"] if isinstance(emo_result, list) else emo_result["emotion"]

        # Eliminar emociones no deseadas
        emociones = {k: v for k, v in emociones_completas.items() if k not in ["sad", "angry", "fear"]}

        # Recalcular dominante solo con las emociones restantes
        dominante = max(emociones, key=emociones.get) if emociones else "neutral"

        # --- BPM: media del canal verde del rostro ---
        y, x, h, w = 50, 50, 100, 100
        roi = frame[y:y+h, x:x+w]
        green = roi[:, :, 1].mean()

        frame_emociones.append({
            "Segundo": segundo,
            "Emoción dominante": dominante,
            "BPM estimado": green,
            **emociones
        })
        bpm_signal.append(green)

    except Exception as e:
        st.warning(f"⚠️ Error en el segundo {segundo}: {e}")

cap.release()

# --- Filtrar señal BPM estimada ---
# --- Procesamiento simple de señal BPM (reescalado visual entre 60 y 100) ---
if len(bpm_signal) > 5:
    bpm_filtered = np.gradient(bpm_signal)
    raw_values = [abs(x) for x in bpm_filtered]

    min_val = min(raw_values)
    max_val = max(raw_values)
    bpm_bpm = [
        60 + 40 * (v - min_val) / (max_val - min_val) if max_val > min_val else 70
        for v in raw_values
    ]
else:
    bpm_bpm = [70] * len(bpm_signal)  # valor neutro si hay pocos datos

# Agregar BPM corregido al dataframe
for i in range(len(frame_emociones)):
    frame_emociones[i]["BPM final"] = round(bpm_bpm[i], 2) if i < len(bpm_bpm) else None

df = pd.DataFrame(frame_emociones)
st.success("✅ Análisis completo.")
st.dataframe(df)

# --- GRAFICAR EMOCIONES DOMINANTES ---
st.subheader("📊 Emoción dominante por segundo")
fig1, ax1 = plt.subplots()
ax1.plot(df["Segundo"], df["Emoción dominante"], marker='o')
ax1.set_xlabel("Segundo")
ax1.set_ylabel("Emoción dominante")
plt.xticks(rotation=90)
st.pyplot(fig1)

# --- GRAFICAR BPM ---
st.subheader("📈 BPM estimado por segundo")
fig2, ax2 = plt.subplots()
ax2.plot(df["Segundo"], df["BPM final"], marker='o', color='green')
ax2.set_xlabel("Segundo")
ax2.set_ylabel("BPM")
st.pyplot(fig2)

# --- Descripción manual por segundo ---
st.subheader("📝 Describe manualmente lo que hace el usuario (por tramos)")
narrativa = st.text_area("Escribe aquí tu descripción", height=200, placeholder="Ej: del 0 al 5 abre la bebida, del 6 al 10 la huele...")

# --- OpenAI GPT Análisis ---
st.subheader("🧠 Análisis estratégico con IA de Neuromarketing")

# Validar que el DataFrame existe
if 'df' not in locals() or df.empty:
    st.error("❌ No se generaron datos para el análisis. Verifica que el video tenga rostro visible.")
    st.stop()

# --- TRANSCRIPCIÓN DE AUDIO ---
import speech_recognition as sr
import moviepy.editor as mp

st.subheader("🗣️ Transcripción automática del audio del video")

# Extraer audio del video a .wav
audio_path = os.path.join(tempfile.gettempdir(), "audio_temp.wav")
try:
    videoclip = mp.VideoFileClip(video_path)
    videoclip.audio.write_audiofile(audio_path, verbose=False, logger=None)
except Exception as e:
    st.error(f"❌ Error extrayendo el audio del video: {e}")
    audio_path = None

# Transcribir audio con SpeechRecognition
transcripcion = ""
if audio_path:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        transcripcion = recognizer.recognize_google(audio_data, language="es-ES")
        st.success("✅ Transcripción completada.")
        st.text_area("📝 Transcripción detectada:", transcripcion, height=150)
    except Exception as e:
        st.warning(f"⚠️ No se pudo transcribir el audio: {e}")

# Crear cliente OpenAI usando API key desde Streamlit Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

try:
    emocion_predominante = df["Emoción dominante"].mode()[0] if not df.empty else "indefinida"
    bpm_promedio = np.mean(df["BPM final"].dropna())
    tabla = df[["Segundo", "Emoción dominante", "BPM final"]].to_string(index=False)

    if narrativa.strip() == "":
        st.warning("⚠️ Escribe una narrativa manual antes de generar el informe.")
        st.stop()

    if transcripcion.strip() == "":
        transcripcion = "(no se detectó audio válido)"

    prompt = f"""
Eres un consultor experto en neuromarketing, comportamiento del consumidor y marketing sensorial. Analiza el siguiente estudio:

- Un usuario interactúa con un producto (por ejemplo, una bebida).
- Se registraron sus emociones por segundo (análisis facial), su BPM (ritmo cardíaco) y lo que dice mientras lo manipula.
- A continuación, se presenta una descripción manual de lo que hizo en el video, una transcripción automática del audio, y una tabla con las emociones y BPM por segundo.

Descripción del comportamiento:
{narrativa}

Transcripción automática del audio:
{transcripcion}

Tabla de datos emocionales y fisiológicos:
{tabla}

Entrega un análisis estructurado con:
1. Resumen emocional por etapa
2. Análisis fisiológico (BPM)
3. Coherencia entre lo que hace, dice y siente
4. Reacciones emocionales específicas ante el producto físico observado: ¿qué partes del objeto (empaque, forma, color, textura) generan respuestas emocionales positivas o negativas?, ¿hay momentos clave de conexión o rechazo?
5. Asociación entre lo que el usuario ve, dice y siente: ¿cómo responde emocionalmente a los estímulos visuales concretos?, ¿hay elementos que provocan sorpresa, agrado, confianza o incomodidad?, ¿qué implicaciones tiene esto para el diseño visual o sensorial?

No repitas la tabla ni la narrativa. Escribe como si entregarás un informe profesional a una marca global.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un experto en neuromarketing y marketing sensorial."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.75,
        max_tokens=1000
    )

    resultado = response.choices[0].message.content
    st.markdown("### 📋 Informe profesional generado por IA:")
    st.markdown(resultado)

except Exception as e:
    st.error(f"❌ Error al generar el análisis con OpenAI: {e}")
