import streamlit as st
import pandas as pd
import datetime as dt
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# --- 2. ESTILOS CSS (CORREGIDO PARA IPHONE/MODO OSCURO) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FONDO GENERAL */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        background-attachment: fixed;
    }
    
    /* CONTENEDOR PRINCIPAL (TARJETA BLANCA) */
    .block-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid #e0e0e0;
    }
    
    /* FUERZA EL COLOR DE TEXTO A OSCURO SIEMPRE (SOLUCIÓN IOS) */
    h1, h2, h3, h4, h5, h6, p, div, span, li {
        color: #333333 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* TÍTULOS ESPECÍFICOS EN VERDE */
    h1 { color: #2e7d32 !important; }
    h3 { color: #388e3c !important; }
    
    /* BOTONES */
    div.stButton > button {
        background: linear-gradient(to right, #43a047, #66bb6a);
        color: white !important; /* Texto del botón blanco */
        border-radius: 10px; border: none;
        padding: 12px 24px; font-size: 16px; font-weight: 600; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
    
    /* CAJAS DE TEXTO PERSONALIZADAS */
    .text-justify { 
        text-align: justify; 
        font-size: 16px; 
        line-height: 1.6;
        color: #424242 !important;
    }
    .explanation-box { 
        background-color: #f1f8e9; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 6px solid #8bc34a; 
        margin-top: 20px;
        color: #1b5e20 !important;
    }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #2196f3;
        margin-bottom: 20px;
        color: #0d47a1 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. RECURSOS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

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

# --- 5. MENÚ ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2454/2454269.png", width=90)
st.sidebar.title("Menú Principal")
menu = st.sidebar.radio(
    "Navegación:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Semestre:** 2025-II\n**Estado:** Sistema Activo 🟢")

# --- 6. APP PRINCIPAL ---
if df is not None:
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    try:
        nums_split = df["Winning Numbers"].str.split(" ", expand=True)
        for i in range(5):
            df[f'Num{i+1}'] = pd.to_numeric(nums_split[i])
    except: pass

    # === INICIO (RENOVADO) ===
    if menu == "🏠 Inicio":
        c1, c2 = st.columns([2, 1])
        with c1:
            st.title("💸 Sistema de Aprendizaje Estadístico: Cash4Life")
            st.markdown("### 🎓 Universidad Privada Antenor Orrego")
            st.markdown("---")
            
            # --- NUEVA INTRODUCCIÓN BASADA EN EL DOCUMENTO ---
            st.markdown("""
            <div class="text-justify">
            <b>1. El Problema de Investigación:</b><br>
            A pesar de que los sorteos de <i>Cash4Life</i> están diseñados bajo principios de aleatoriedad, surge la interrogante científica: 
            ¿Realmente se distribuyen los números de forma uniforme o existen patrones ocultos y sesgos temporales que pasan desapercibidos?
            <br><br>
            <b>2. Nuestra Solución Tecnológica:</b><br>
            Hemos desarrollado un sistema inteligente que procesa miles de registros históricos (2014-Presente). Utilizando algoritmos de 
            <b>Machine Learning (Regresión Lineal y Árboles de Decisión)</b>, el sistema desafía al azar buscando correlaciones matemáticas 
            entre la fecha del sorteo y los números ganadores.
            <br><br>
            <b>3. Hallazgos Clave:</b><br>
            Los resultados obtenidos validan la integridad del juego. La baja capacidad predictiva de los modelos confirma que la lotería 
            se comporta como un sistema estocástico (aleatorio) robusto, donde el pasado no predice el futuro.
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with c2:
            if lottie_robot_intro: st_lottie(lottie_robot_intro, height=350)
            with st.expander("👨‍💻 Ver Equipo de Investigación"):
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

    # === ANÁLISIS ===
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Exploración de Datos")
        tab1, tab2 = st.tabs(["📄 Base de Datos", "📈 Frecuencias"])
        
        with tab1:
            df_vis = df.copy()
            df_vis['Draw Date'] = df_vis['Draw Date'].dt.strftime('%Y-%m-%d')
            cols = ['Draw Date', 'Winning Numbers', 'Cash Ball', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5']
            st.dataframe(df_vis[cols], use_container_width=True, height=400)
        
        with tab2:
            st.subheader("🏆 Números Más Frecuentes")
            all_numbers = pd.concat([df[f'Num{i}'] for i in range(1, 6)])
            freq_counts = all_numbers.value_counts().head(10)
            col_chart, col_table = st.columns([2, 1])
            with col_chart: st.bar_chart(freq_counts, color="#4CAF50")
            with col_table: st.dataframe(freq_counts, use_container_width=True)

    # === PREDICCIÓN (REGRESIÓN) ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia (Regresión)")
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        with c_input:
            fecha_input = st.date_input("Fecha Objetivo:", dt.date.today() + dt.timedelta(days=1))
            predict_btn = st.button("🚀 Ejecutar Modelo Predictivo")
            
        with c_anim:
            anim_placeholder = st.empty()
            
        if predict_btn:
            with c_anim:
                if lottie_calculating: st_lottie(lottie_calculating, height=150, key="calc")
            with st.spinner("Calculando regresión..."):
                time.sleep(1.5)

            pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = max(1, min(60, int(round(pred_val))))
            resto = np.sort(np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False))
            
            st.markdown("---")
            st.subheader(f"🎫 Ticket Generado")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2", resto[0])
            b3.metric("Bola 3", resto[1])
            b4.metric("Bola 4", resto[2])
            b5.metric("Bola 5", resto[3])

            st.markdown("### 📊 Análisis de Resultados")
            
            tab_graph, tab_error = st.tabs(["📉 Tendencia", "📋 Tabla de Error (Explicada)"])
            
            with tab_graph:
                fig, ax = plt.subplots(figsize=(10, 4))
                sample = df.sample(min(500, len(df)))
                ax.scatter(sample['Draw Date'], sample['Num1'], color='#2196F3', alpha=0.4, label='Datos Reales')
                
                date_range = np.array([X.min(), X.max()]).reshape(-1, 1)
                pred_line = model.predict(date_range)
                ax.plot([df['Draw Date'].min(), df['Draw Date'].max()], pred_line, color='red', linewidth=3, label='Predicción IA')
                
                ax.set_ylabel("Valor Bola 1")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.caption(f"R² = {r2:.5f} (Tendencia casi nula, confirmando aleatoriedad).")

            with tab_error:
                # Generar tabla de error
                last_5 = df.tail(5).copy()
                last_5['Draw Date'] = last_5['Draw Date'].dt.strftime('%Y-%m-%d')
                last_5['Predicción IA'] = model.predict(last_5[['DrawDate_Ordinal']]).round().astype(int)
                last_5['Diferencia (Error)'] = abs(last_5['Num1'] - last_5['Predicción IA'])
                
                st.write("**Comparativa Reciente: Realidad vs. Modelo**")
                st.dataframe(last_5[['Draw Date', 'Num1', 'Predicción IA', 'Diferencia (Error)']], use_container_width=True)
                
                # --- EXPLICACIÓN DETALLADA QUE PEDISTE ---
                st.markdown("""
                <div class="explanation-box">
                <b>📘 ¿Cómo leer esta Tabla de Error?</b><br><br>
                <b>1. El Concepto:</b> Esta tabla compara el número que <i>realmente salió</i> (Num1) contra lo que la <i>IA calculó</i> que saldría.<br>
                <b>2. La Columna 'Diferencia':</b> Es la distancia entre ambos números. Por ejemplo, si salió el <b>60</b> y la IA predijo <b>10</b>, la diferencia es <b>50</b>.<br><br>
                <b>💡 ¿Por qué hay tanto error?</b><br>
                En este proyecto, <b>un error alto es un resultado científicamente correcto</b>. Significa que los números saltan aleatoriamente lejos del promedio, 
                demostrando que el sorteo NO está trucado y es imposible de predecir con una línea recta. Si el error fuera 0, significaría que la lotería está manipulada.
                </div>
                """, unsafe_allow_html=True)

    # === CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 Clasificación Cash Ball")
        
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        st.write("##### Ingrese la combinación:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Calcular Probabilidad"):
            input_data = [[n1,n2,n3,n4,n5]]
            probs = clf.predict_proba(input_data)[0]
            pred_class = clf.predict(input_data)[0]
            
            st.balloons()
            st.success(f"La Cash Ball más probable es: **{pred_class}**")
            
            st.markdown("### 📊 Desglose de Probabilidades")
            col_prob, col_desc = st.columns([2, 1])
            
            with col_prob:
                prob_df = pd.DataFrame({'Cash Ball': [1, 2, 3, 4], 'Probabilidad (%)': probs * 100})
                st.bar_chart(prob_df.set_index('Cash Ball'), color="#2196F3")
            
            with col_desc:
                st.markdown(f"""
                <div class="highlight-box">
                <b>Confianza del Modelo:</b><br>
                Existe un <b>{probs[pred_class-1]*100:.1f}%</b> de probabilidad matemática de que salga el {pred_class}, 
                basado en patrones históricos similares.
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")



