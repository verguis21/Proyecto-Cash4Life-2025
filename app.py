import streamlit as st
import pandas as pd
import datetime as dt
import requests
import time
import numpy as np
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# CSS Personalizado para mejorar la apariencia
st.markdown("""
<style>
    /* Ocultar menú de hamburguesa y footer de Streamlit para look más 'App' */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Estilo personalizado para métricas */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00C853; /* Verde Dinero */
    }
    
    /* Botones personalizados */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition-duration: 0.4s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
        border: 2px solid white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES UTILITARIAS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Cargar animaciones
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_lottery = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json")
lottie_robot = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")

@st.cache_data
def load_data():
    file_path = "Lottery_Cash_4_Life_Winning_Numbers__Beginning_2014.csv"
    try:
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        return df
    except FileNotFoundError: return None

df = load_data()

# --- 3. BARRA LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1086/1086581.png", width=80)
st.sidebar.title("Menú Principal")
menu = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📊 Análisis de Datos", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto Universitario - 2025")

# --- 4. LÓGICA PRINCIPAL ---
if df is not None:
    # Preprocesamiento
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    try:
        nums = df["Winning Numbers"].str.split(" ", expand=True)
        for i in range(5):
            df[f'Num{i+1}'] = pd.to_numeric(nums[i])
    except: pass

    # --- MÓDULO INICIO ---
    if menu == "🏠 Inicio":
        c1, c2 = st.columns([1, 2])
        with c1:
            if lottie_robot: st_lottie(lottie_robot, height=280, key="bot")
        with c2:
            st.title("Sistema Inteligente Cash4Life")
            st.markdown("#### Universidad Privada Antenor Orrego")
            st.info("Bienvenido al sistema de análisis predictivo basado en Machine Learning.")
            
            st.markdown("### 📅 Estado del Sorteo")
            # Lógica de Próximo Sorteo (Cash4Life es Diario)
            hoy = dt.date.today()
            prox_sorteo = hoy + dt.timedelta(days=1)
            st.success(f"✅ Próximo Sorteo Oficial: **Mañana, {prox_sorteo.strftime('%d de %B de %Y')}**")

    # --- MÓDULO ANÁLISIS ---
    elif menu == "📊 Análisis de Datos":
        st.title("Exploración de Datos")
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("Visualización de los últimos registros ingresados al sistema.")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.metric("Total Datos", len(df))
            if lottie_analysis: st_lottie(lottie_analysis, height=150, key="ana")

    # --- MÓDULO PREDICCIÓN (REGRESIÓN) ---
    elif menu == "🔮 Predicción (Regresión)":
        st.title("🔮 Predicción de Tendencia")
        st.markdown("Modelo: **Regresión Lineal Simple** | Objetivo: Predecir el *Primer Número*.")
        
        # Entrenar modelo
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        col_izq, col_der = st.columns([2,1])
        
        with col_izq:
            st.markdown("### 🗓️ Configuración del Sorteo")
            
            # Cálculo automático de la próxima fecha lógica
            tomorrow = dt.date.today() + dt.timedelta(days=1)
            
            # Mostrar fecha sugerida visualmente
            st.info(f"💡 Fecha sugerida para el próximo sorteo: **{tomorrow}**")
            
            fecha_input = st.date_input("Seleccione fecha a predecir:", tomorrow)
            
            if st.button("🎰 Generar Ticket Predictivo"):
                # Efecto de carga
                with st.spinner("Calculando probabilidades matemáticas..."):
                    time.sleep(1.5)
                
                # Predicción
                pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
                n1 = int(round(pred_val))
                n1 = max(1, min(60, n1)) # Limitar entre 1 y 60
                
                # Simulación del resto (Visual)
                resto = np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False)
                resto.sort()
                
                st.markdown("---")
                st.subheader(f"🎫 Ticket Probable para el {fecha_input}")
                
                # Mostrar bolas bonitas
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Bola 1 (IA)", n1)
                b2.metric("Bola 2", resto[0])
                b3.metric("Bola 3", resto[1])
                b4.metric("Bola 4", resto[2])
                b5.metric("Bola 5", resto[3])
                
                st.caption(f"Confianza estadística del modelo (R²): {r2:.5f}")

        with col_der:
            if lottie_lottery: st_lottie(lottie_lottery, height=250, key="loto")

    # --- MÓDULO CLASIFICACIÓN ---
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.title("🟢 Predicción Cash Ball")
        st.markdown("Modelo: **Árbol de Decisión** | Objetivo: Predecir la *Bola Extra*.")
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        st.write("Ingrese la combinación ganadora principal:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Calcular Cash Ball"):
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            st.balloons()
            st.success(f"La Cash Ball más probable es: **{pred}**")

else:
    st.error("⚠️ Error: Archivo CSV no encontrado.")
