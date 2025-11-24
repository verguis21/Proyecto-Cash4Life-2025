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

# --- 2. ESTILOS CSS AVANZADOS (FONDO + BOTONES) ---
# URL de la imagen de fondo (Dinero cayendo / abstracto)
background_url = "https://img.freepik.com/free-vector/green-money-background-with-falling-banknotes_1017-30248.jpg"

st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* FONDO DE PANTALLA PRINCIPAL */
    [data-testid="stAppViewContainer"] {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* CAPA SEMITRANSPARENTE PARA QUE SE LEA EL TEXTO */
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    
    /* CONTENEDOR PRINCIPAL CON FONDO BLANCO SUAVE */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.92); /* Blanco al 92% de opacidad */
        border-radius: 20px;
        padding: 3rem;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}

    /* Estilo de Métricas */
    div[data-testid="stMetricValue"] {{ font-size: 26px; color: #008000; font-weight: bold; }}
    
    /* Botones */
    div.stButton > button {{
        background-color: #006400; color: white; border-radius: 12px; border: none;
        padding: 12px 28px; font-size: 16px; font-weight: 600; transition: 0.3s;
        width: 100%;
    }}
    div.stButton > button:hover {{ background-color: #008000; transform: scale(1.02); }}
    
    /* Texto Intro */
    .intro-text {{ font-size: 18px; color: #333; text-align: justify; line-height: 1.6; }}
</style>
""", unsafe_allow_html=True)

# --- 3. CARGA DE ANIMACIONES (LOTTIE) ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Nuevas animaciones
lottie_robot_intro = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")
lottie_calculating = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json") # Robot procesando
lottie_money = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json") # Dinero

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
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2454/2454269.png", width=100)
st.sidebar.title("Menú Cash4Life")
menu = st.sidebar.radio(
    "Seleccione Opción:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Universidad Privada Antenor Orrego**\nIngeniería de Sistemas e IA")

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
            st.markdown("### Proyecto de Aprendizaje Estadístico")
            st.markdown("---")
            st.markdown("""
            <div class="intro-text">
            Bienvenido. Este sistema utiliza algoritmos avanzados de <b>Machine Learning</b> para desafiar 
            la aleatoriedad de la lotería Cash4Life de Nueva York. Analizamos miles de sorteos históricos 
            (2014-Presente) buscando patrones ocultos.
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            c1, c2 = st.columns(2)
            c1.success("📈 **Regresión:** Predicción de tendencias.")
            c2.info("🤖 **Clasificación:** IA para la Cash Ball.")
            
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with col_anim:
            if lottie_robot_intro: st_lottie(lottie_robot_intro, height=350)
            
            with st.expander("👨‍💻 Ver Autores"):
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
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Exploración de la Data")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(df.head(15), use_container_width=True)
        with c2:
            st.metric("Total Sorteos", f"{len(df):,}")
            st.metric("Años Analizados", f"{2014} - {dt.date.today().year}")

    # === PESTAÑA PREDICCIÓN (CON ROBOT NUEVO) ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia (Regresión)")
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        
        with c_input:
            st.markdown("### Configurar Predicción")
            tomorrow = dt.date.today() + dt.timedelta(days=1)
            fecha_input = st.date_input("Fecha del Sorteo:", tomorrow)
            
            predict_btn = st.button("🚀 Iniciar Cálculo Predictivo")
            
        with c_anim:
            # Espacio reservado para la animación
            anim_placeholder = st.empty()
            
        if predict_btn:
            # MOSTRAR ROBOT CALCULANDO
            with c_anim:
                if lottie_calculating: 
                    st_lottie(lottie_calculating, height=200, key="calc")
            
            with st.spinner("El modelo está procesando algoritmos matemáticos..."):
                time.sleep(3) # Tiempo para ver la animación
            
            # Cálculo real
            pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = int(round(pred_val))
            n1 = max(1, min(60, n1))
            
            resto = np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False)
            resto.sort()
            
            st.markdown("---")
            st.subheader(f"🎫 Ticket Generado por la IA")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2", resto[0])
            b3.metric("Bola 3", resto[1])
            b4.metric("Bola 4", resto[2])
            b5.metric("Bola 5", resto[3])
            
            st.caption(f"Confianza del Modelo (R²): {r2:.5f}")
            st.success("¡Cálculo finalizado exitosamente!")

    # === PESTAÑA CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 IA: Clasificación Cash Ball")
        st.markdown("Ingrese los 5 números ganadores para predecir la bola extra.")
        
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
        
        if st.button("🎱 Predecir Cash Ball"):
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            st.balloons()
            st.metric("Cash Ball Probable", pred)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")




