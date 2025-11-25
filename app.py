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

# CONFIGURACIÓN DE PÁGINA 
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# ESTILOS CSS 
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        background-attachment: fixed;
    }
    
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .block-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    h1 { color: #2e7d32; font-family: 'Helvetica', sans-serif; }
    h3 { color: #388e3c; }
    
    div.stButton > button {
        background: linear-gradient(to right, #43a047, #66bb6a);
        color: white; border-radius: 10px; border: none;
        padding: 12px 24px; font-size: 16px; font-weight: 600; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%;
    }
    
    .text-justify { text-align: justify; font-size: 17px; color: #424242; line-height: 1.6; }
    .highlight { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #43a047; }
</style>
""", unsafe_allow_html=True)

#  RECURSOS 
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_robot_intro = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")
lottie_calculating = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")

#   CARGA DE DATOS 
@st.cache_data
def load_data():
    file_path = "Lottery_Cash_4_Life_Winning_Numbers__Beginning_2014.csv"
    try:
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        return df
    except FileNotFoundError: return None

df = load_data()

#  MENÚ 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2454/2454269.png", width=90)
st.sidebar.title("Menú Principal")
menu = st.sidebar.radio(
    "Navegación:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Semestre:** 2025-II\n**Estado:** Sistema Activo 🟢")

# APP PRINCIPAL 
if df is not None:
    # Preprocesamiento general
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    try:
        nums_split = df["Winning Numbers"].str.split(" ", expand=True)
        cols_nums = []
        for i in range(5):
            col_name = f'Num{i+1}'
            df[col_name] = pd.to_numeric(nums_split[i])
            cols_nums.append(col_name)
    except: pass

    #  INICIO 
    if menu == "🏠 Inicio":
        col_text, col_anim = st.columns([2, 1])
        with col_text:
            st.title("💸 Sistema Predictivo Cash4Life")
            st.markdown("### 🎓 Proyecto de Aprendizaje Estadístico")
            st.markdown("---")
            
            st.markdown("""
            <div class="text-justify">
            Bienvenido a la plataforma de análisis inteligente de loterías. Este proyecto nace de la necesidad de aplicar 
            conceptos teóricos de <b>Estadística y Machine Learning</b> sobre un escenario real y complejo: el azar.
            <br><br>
            Analizamos miles de sorteos históricos (2014-Presente) de la lotería de Nueva York para responder una pregunta clave:
            <i>¿Es posible identificar patrones matemáticos en un sistema diseñado para ser aleatorio?</i>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown('<div class="highlight"><b>🛠️ Metodología Aplicada:</b><br>Utilizamos la librería <i>Scikit-Learn</i> de Python para entrenar modelos supervisados con datos históricos, buscando minimizar el error cuadrático medio (MSE) en las predicciones.</div>', unsafe_allow_html=True)

            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with col_anim:
            if lottie_robot_intro: st_lottie(lottie_robot_intro, height=320)
            with st.expander("👨‍💻 Equipo de Investigación"):
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

    # ANÁLISIS HISTÓRICO
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Exploración de Datos")
        st.markdown("""
        <div class="text-justify">
        En esta sección realizamos la Minería de Datos. Analizamos la distribución de los números ganadores para detectar
        qué bolas tienen mayor frecuencia de aparición ("Números Calientes").
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        tab1, tab2 = st.tabs(["📄 Base de Datos Completa", "📈 Análisis de Frecuencia"])
        
        with tab1:
            c1, c2 = st.columns([3, 1])
            with c1:
                # Crear copia para visualización
                df_vis = df.copy()
                
                df_vis['Draw Date'] = df_vis['Draw Date'].dt.strftime('%Y-%m-%d')
                
                
                cols_to_show = ['Draw Date', 'Winning Numbers', 'Cash Ball', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5']
                
                st.dataframe(df_vis[cols_to_show], use_container_width=True, height=400)
            with c2:
                st.metric("Total Registros", f"{len(df):,}")
                st.info("ℹ️ Dataset estático actualizado al periodo 2025.")

        with tab2:
            st.subheader("🏆 Números Más Frecuentes")
            all_numbers = pd.concat([df[f'Num{i}'] for i in range(1, 6)])
            freq_counts = all_numbers.value_counts().head(10)
            
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                st.bar_chart(freq_counts, color="#4CAF50")
            with col_table:
                st.write("**Top 10 Números:**")
                st.dataframe(freq_counts, use_container_width=True)

    # PREDICCIÓN (REGRESIÓN)
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia")
        
        with st.expander("📘 ¿Cómo funciona este modelo? (Explicación Técnica)"):
            st.markdown("""
            Utilizamos un modelo de **Regresión Lineal Simple**.
            1. Convertimos la fecha del sorteo a un número ordinal.
            2. Entrenamos el modelo (`model.fit`) para encontrar una línea recta que minimice la distancia entre la fecha y el primer número ganador.
            3. El resultado nos muestra la tendencia central del sorteo.
            """)
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        with c_input:
            tomorrow = dt.date.today() + dt.timedelta(days=1)
            st.write("##### Configuración de Simulación:")
            fecha_input = st.date_input("Fecha Objetivo:", tomorrow)
            predict_btn = st.button("🚀 Ejecutar Modelo Predictivo")
            
        with c_anim:
            anim_placeholder = st.empty()
            
        if predict_btn:
            with c_anim:
                if lottie_calculating: st_lottie(lottie_calculating, height=180, key="calc")
            
            with st.spinner("La IA está calculando probabilidades..."):
                time.sleep(2)
            
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
            
            st.caption(f"R² (Coeficiente de Determinación): {r2:.5f}")
            st.info("💡 **Conclusión del Modelo:** La baja correlación (R² cercano a 0) valida la hipótesis nula: los sorteos son eventos independientes y aleatorios.")

    #  CLASIFICACIÓN
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 IA: Clasificación Cash Ball")
        
        with st.expander("📘 ¿Cómo funciona este modelo? (Explicación Técnica)"):
            st.markdown("""
            Utilizamos un algoritmo de **Árbol de Decisión (Decision Tree Classifier)**.
            * El modelo analiza las combinaciones de los 5 números principales históricos.
            * Aprende qué 'Cash Ball' (1, 2, 3 o 4) suele aparecer con ciertos patrones numéricos.
            * Al ingresar nuevos números, el árbol recorre sus ramas para sugerir la clase más probable.
            """)
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        st.write("##### Ingrese la combinación ganadora principal:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Calcular Probabilidad"):
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            st.balloons()
            st.metric("Cash Ball Probable", pred)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")


