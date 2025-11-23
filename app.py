import streamlit as st
import pandas as pd
import datetime as dt
import requests
import time
import numpy as np
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# --- FUNCIÓN PARA CARGAR ANIMACIONES LOTTIE ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Cargar animaciones (URLs públicas de LottieFiles)
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_lottery = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json")
lottie_robot = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    file_path = "Lottery_Cash_4_Life_Winning_Numbers__Beginning_2014.csv"
    try:
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

# --- BARRA LATERAL ---
st.sidebar.title("🎛️ Panel de Control")
menu = st.sidebar.radio(
    "Navegación:",
    ["🏠 Inicio", "📊 Análisis de Datos", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("v2.0 - Edición Proyecto Final")

if df is not None:
    # Preprocesamiento
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    try:
        nums = df["Winning Numbers"].str.split(" ", expand=True)
        for i in range(5):
            df[f'Num{i+1}'] = pd.to_numeric(nums[i])
    except:
        pass

    # --- 1. INICIO ---
    if menu == "🏠 Inicio":
        col1, col2 = st.columns([1, 2])
        with col1:
            if lottie_robot:
                st_lottie(lottie_robot, height=300, key="robot")
        with col2:
            st.title("Sistema Inteligente Cash4Life")
            st.markdown("### Universidad Privada Antenor Orrego")
            st.success("Bienvenido al sistema de análisis predictivo basado en Machine Learning.")
            st.markdown("""
            Este software permite:
            * 🕵️‍♀️ **Explorar** patrones históricos ocultos.
            * 📈 **Predecir** tendencias usando Regresión Lineal.
            * 🧠 **Clasificar** resultados probables con IA.
            """)

    # --- PESTAÑA 2: ANÁLISIS ---
    elif menu == "Análisis de Datos":
        st.title("📊 Exploración de Datos Históricos")
        st.markdown("""
        En esta sección se observan los registros 'crudos' obtenidos de la Lotería de Nueva York.
        Sirve para verificar la integridad de los datos antes de procesarlos.
        """)
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Sorteos Registrados", len(df))
        with col2:
            st.metric("Rango de Fechas", f"{df['Draw Date'].dt.year.min()} - {df['Draw Date'].dt.year.max()}")

    # --- PESTAÑA 3: REGRESIÓN (Corregido a Enteros) ---
    elif menu == "Predicción (Regresión)":
        st.title("📈 Modelo de Regresión Lineal")
        st.markdown("""
        **Objetivo:** Intentar predecir el valor del **Primer Número Ganador (Num1)** basándose únicamente en la fecha del sorteo.
        _Nota: Un resultado lejano a la realidad confirma la aleatoriedad del juego._
        """)
        
        # Lógica del modelo
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        r2 = r2_score(y_test, model.predict(X_test))
        st.metric("Precisión del Modelo (R²)", f"{r2:.4f}")
        
        st.markdown("---")
        st.subheader("🔮 Simular Predicción")
        fecha = st.date_input("Seleccione una fecha futura para el sorteo:")
        
        if st.button("Predecir Primer Número"):
            pred_float = model.predict([[dt.datetime.toordinal(fecha)]])
            # AQUÍ ESTÁ LA MAGIA: int(round(...)) convierte decimal a entero
            pred_entero = int(round(pred_float[0]))
            
            # Evitar que prediga números negativos o cero (por lógica de lotería)
            if pred_entero < 1: pred_entero = 1
            
            st.success(f"Según la tendencia histórica, el modelo predice que el primer número sería: **{pred_entero}**")

    # --- PESTAÑA 4: CLASIFICACIÓN ---
    elif menu == "Clasificación (Cash Ball)":
        st.title("🟢 Clasificación de Cash Ball")
        st.markdown("""
        **Objetivo:** Predecir el número especial **'Cash Ball'** (del 1 al 4) utilizando los 5 números principales ganadores.
        Este módulo utiliza un algoritmo de **Árbol de Decisión**.
        """)
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        
        st.markdown("---")
        st.subheader("🔢 Ingrese los números ganadores:")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("Bola 1", 1, 60, 5)
        n2 = c2.number_input("Bola 2", 1, 60, 10)
        n3 = c3.number_input("Bola 3", 1, 60, 25)
        n4 = c4.number_input("Bola 4", 1, 60, 30)
        n5 = c5.number_input("Bola 5", 1, 60, 45)
        
        if st.button("Calcular Cash Ball Probable"):
            pred = model.predict([[n1,n2,n3,n4,n5]])
            st.balloons()
            st.success(f"La Cash Ball predicha por el patrón es: **{pred[0]}**")

else:
    st.error("⚠️ Error: No se encontró el archivo CSV en el repositorio.")

