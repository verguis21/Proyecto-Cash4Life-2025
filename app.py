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

# CSS CORREGIDO (ADAPTATIVO)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Métricas en Verde Dinero */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00C853; }
    
    /* Botones Estilizados */
    div.stButton > button {
        background-color: #4CAF50; color: white; border-radius: 10px; border: none;
        padding: 10px 24px; font-size: 16px; transition-duration: 0.4s;
    }
    div.stButton > button:hover { background-color: #45a049; border: 2px solid white; }
    
    /* Texto de Introducción Inteligente (Se adapta al tema) */
    .intro-text { 
        font-size: 18px; 
        font-weight: 500; 
        line-height: 1.6;
        text-align: justify;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CARGA DE RECURSOS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

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

# --- 3. MENÚ LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1086/1086581.png", width=80)
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📊 Análisis de Datos", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Curso:** Aprendizaje Estadístico\n**Semestre:** 2025-1")

# --- 4. LÓGICA PRINCIPAL ---
if df is not None:
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    try:
        nums = df["Winning Numbers"].str.split(" ", expand=True)
        for i in range(5):
            df[f'Num{i+1}'] = pd.to_numeric(nums[i])
    except: pass

    # === PESTAÑA INICIO ===
    if menu == "🏠 Inicio":
        col_text, col_anim = st.columns([2, 1])
        
        with col_text:
            st.title("Sistema de Aprendizaje Estadístico: Cash4Life")
            st.markdown("### 🏛️ Universidad Privada Antenor Orrego")
            st.markdown("---")
            
            # TEXTO CORREGIDO (Sin color fijo)
            st.markdown("""
            <div class="intro-text">
            Este proyecto desarrolla un análisis profundo sobre los sorteos de la lotería 
            <b>Cash4Life (New York)</b>. A pesar de ser un juego de azar diseñado bajo principios 
            de aleatoriedad, esta investigación busca identificar posibles <b>patrones estadísticos, 
            sesgos o tendencias ocultas</b> en los datos históricos.
            <br><br>
            Utilizando algoritmos de <b>Machine Learning</b>, el sistema permite:
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Espacio
            c1, c2 = st.columns(2)
            c1.info("📈 **Regresión Lineal:**\nAnalizar si el paso del tiempo influye en los números ganadores.")
            c2.success("🤖 **Clasificación (IA):**\nPredecir la 'Cash Ball' usando Árboles de Decisión.")
            
            # Próximo Sorteo
            hoy = dt.date.today()
            manana = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {manana.strftime('%d de %B de %Y')}")

        with col_anim:
            if lottie_robot: st_lottie(lottie_robot, height=400, key="bot_intro")
            
            with st.expander("👥 Ver Equipo de Investigación"):
                st.write("""
                * Bernabé Arce, James Franco
                * Coronado Medina, Sergio Adrian
                * Enriquez Cabanillas, César
                * Carrascal Carranza, Hetzer
                * Lázaro Velásquez, Jesús Alberto
                * Martino López, Marielsys Paola
                * Mori Galarza, Franco
                * Vergaray Colonia, José Francisco
                """)

    # === PESTAÑA ANÁLISIS ===
    elif menu == "📊 Análisis de Datos":
        st.title("📊 Exploración de Datos Históricos")
        st.markdown("Visualización de la integridad y distribución de los datos recolectados (2014-Presente).")
        
        col1, col2 = st.columns([3,1])
        with col1:
            st.dataframe(df.head(15), use_container_width=True)
        with col2:
            st.metric("Total de Sorteos", f"{len(df):,}")
            st.metric("Variables Analizadas", "7 (Fecha + 6 Bolas)")
            if lottie_analysis: st_lottie(lottie_analysis, height=120, key="ana")

    # === PESTAÑA PREDICCIÓN (REGRESIÓN) ===
    elif menu == "🔮 Predicción (Regresión)":
        st.title("🔮 Modelo de Tendencia Temporal")
        st.markdown("Algoritmo: **Regresión Lineal Simple** | Variable Objetivo: **Primer Número (Num1)**")
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        col1, col2 = st.columns([2,1])
        with col1:
            tomorrow = dt.date.today() + dt.timedelta(days=1)
            fecha_input = st.date_input("Seleccione fecha a analizar:", tomorrow)
            
            if st.button("🎰 Generar Predicción del Ticket"):
                with st.spinner("Procesando modelo matemático..."):
                    time.sleep(1)
                    
                pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
                n1 = int(round(pred_val))
                n1 = max(1, min(60, n1))
                
                # Simulación visual
                resto = np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False)
                resto.sort()
                
                st.markdown("### 🎫 Ticket Probable (IA + Simulación)")
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Bola 1 (Predicha)", n1)
                b2.metric("Bola 2", resto[0])
                b3.metric("Bola 3", resto[1])
                b4.metric("Bola 4", resto[2])
                b5.metric("Bola 5", resto[3])
                
                st.caption(f"Nota: El R² del modelo es {r2:.5f}, lo que confirma la alta aleatoriedad del sorteo.")

        with col2:
            if lottie_lottery: st_lottie(lottie_lottery, height=200, key="loto")

    # === PESTAÑA CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.title("🟢 Predicción de Cash Ball")
        st.markdown("Algoritmo: **Árbol de Decisión** | Objetivo: Clasificar la **Bola Extra** (1-4)")
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        st.write("Ingrese una combinación de 5 números principales:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Predecir Cash Ball"):
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            st.balloons()
            st.success(f"Según el patrón histórico, la Cash Ball debería ser: **{pred}**")

else:
    st.error("⚠️ Error Crítico: No se encontró el dataset en el repositorio.")
