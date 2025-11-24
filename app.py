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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# --- 2. ESTILOS CSS (LIMPIEZA VISUAL) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FONDO DE PANTALLA: DEGRADADO SUAVE (NO MOLESTA A LA VISTA) */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); /* Tonos verdes muy suaves */
        background-attachment: fixed;
    }
    
    /* CAPA SEMITRANSPARENTE */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* CONTENEDOR PRINCIPAL (TARJETA FLOTANTE) */
    .block-container {
        background-color: #ffffff;
        border-radius: 25px;
        padding: 3rem;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); /* Sombra elegante */
        border: 1px solid #e0e0e0;
    }

    /* TÍTULOS */
    h1 { color: #2e7d32; font-family: 'Helvetica', sans-serif; }
    h2, h3 { color: #388e3c; }

    /* ESTILO DE MÉTRICAS */
    div[data-testid="stMetricValue"] { font-size: 26px; color: #1b5e20; font-weight: bold; }
    
    /* BOTONES MODERNOS */
    div.stButton > button {
        background: linear-gradient(to right, #43a047, #66bb6a);
        color: white; 
        border-radius: 12px; 
        border: none;
        padding: 12px 28px; 
        font-size: 16px; 
        font-weight: 600; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    /* TEXTO INTRODUCTORIO */
    .intro-text { 
        font-size: 18px; 
        color: #424242; 
        text-align: justify; 
        line-height: 1.6; 
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CARGA DE ANIMACIONES ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Animaciones (El dinamismo que pediste)
lottie_robot_intro = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")
lottie_calculating = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")

# --- 4. CARGA DE DATOS ---
@st.cache_data
def load_data():
    file_path = "Lottery_Cash_4_Life_Winning_Numbers__Beginning_2014.csv"
    try:
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        return df
    except FileNotFoundError: return None

df = load_data()

# --- 5. MENÚ LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2454/2454269.png", width=90)
st.sidebar.title("Menú Principal")
menu = st.sidebar.radio(
    "Navegación:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.success("**Estado:** Sistema Activo 🟢")

# --- 6. LÓGICA PRINCIPAL ---
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
            st.title("💸 Sistema Predictivo Cash4Life")
            st.markdown("### 🎓 Proyecto de Aprendizaje Estadístico")
            st.markdown("---")
            st.markdown("""
            <div class="intro-text">
            Bienvenido a la plataforma de análisis inteligente. Hemos procesado miles de sorteos históricos 
            (2014-Presente) utilizando algoritmos de <b>Machine Learning</b> para identificar patrones matemáticos 
            en la lotería de Nueva York.
            <br><br>
            Este sistema no garantiza premios, pero utiliza la ciencia de datos para desafiar la aleatoriedad pura.
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            c1, c2 = st.columns(2)
            c1.info("📈 **Regresión Lineal:**\nDetecta tendencias temporales.")
            c2.success("🤖 **Clasificación IA:**\nCalcula probabilidad de Cash Ball.")
            
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with col_anim:
            if lottie_robot_intro: st_lottie(lottie_robot_intro, height=320)
            
            with st.expander("👨‍💻 Ver Autores del Proyecto"):
                st.write("""
                **Universidad Privada Antenor Orrego**
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
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Base de Datos Histórica")
        st.markdown("Exploración de la integridad de los datos recolectados.")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(df.head(15), use_container_width=True)
        with c2:
            st.metric("Registros Totales", f"{len(df):,}")
            st.metric("Variables", "7")

    # === PESTAÑA PREDICCIÓN ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia")
        st.markdown("Modelo: **Regresión Lineal** | Objetivo: **Predecir Ticket**")
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        
        with c_input:
            st.markdown("### ⚙️ Configuración")
            tomorrow = dt.date.today() + dt.timedelta(days=1)
            fecha_input = st.date_input("Fecha del Sorteo:", tomorrow)
            
            predict_btn = st.button("🚀 Ejecutar Modelo Predictivo")
            
        with c_anim:
            anim_placeholder = st.empty()
            
        if predict_btn:
            with c_anim:
                if lottie_calculating: 
                    st_lottie(lottie_calculating, height=180, key="calc")
            
            with st.spinner("Procesando algoritmos..."):
                time.sleep(2.5) 
            
            pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = int(round(pred_val))
            n1 = max(1, min(60, n1))
            
            resto = np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False)
            resto.sort()
            
            st.markdown("---")
            st.subheader(f"🎫 Ticket Generado")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2", resto[0])
            b3.metric("Bola 3", resto[1])
            b4.metric("Bola 4", resto[2])
            b5.metric("Bola 5", resto[3])
            
            st.caption(f"Confianza Estadística (R²): {r2:.5f}")
            st.success("Predicción Finalizada")

    # === PESTAÑA CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 IA: Clasificación Cash Ball")
        st.markdown("Ingrese los 5 números principales para calcular la bola extra.")
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("Bola 1", 1, 60, 5)
        n2 = c2.number_input("Bola 2", 1, 60, 10)
        n3 = c3.number_input("Bola 3", 1, 60, 25)
        n4 = c4.number_input("Bola 4", 1, 60, 30)
        n5 = c5.number_input("Bola 5", 1, 60, 45)
        
        if st.button("🎱 Calcular Probabilidad"):
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            st.balloons()
            st.metric("Cash Ball Probable", pred)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")
