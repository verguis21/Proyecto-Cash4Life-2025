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

# --- 2. ESTILOS CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        background-attachment: fixed;
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
        transition: transform 0.2s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
    .text-justify { text-align: justify; font-size: 16px; color: #424242; line-height: 1.6; }
    .explanation-box { background-color: #f1f8e9; padding: 15px; border-radius: 10px; border-left: 5px solid #8bc34a; margin-top: 20px; }
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

    # === INICIO ===
    if menu == "🏠 Inicio":
        c1, c2 = st.columns([2, 1])
        with c1:
            st.title("💸 Sistema Predictivo Cash4Life")
            st.markdown("### 🎓 Proyecto de Aprendizaje Estadístico")
            st.markdown("---")
            st.markdown("""
            <div class="text-justify">
            Bienvenido. Este sistema utiliza <b>Machine Learning</b> para analizar la lotería Cash4Life (New York).
            Nuestro objetivo es aplicar modelos matemáticos rigurosos para determinar si existe predictibilidad en el azar.
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with c2:
            if lottie_robot_intro: st_lottie(lottie_robot_intro, height=300)

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

    # === PREDICCIÓN (REGRESIÓN) CON GRÁFICOS ===
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

            # Cálculo
            pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = max(1, min(60, int(round(pred_val))))
            resto = np.sort(np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False))
            
            # --- RESULTADOS ---
            st.markdown("---")
            st.subheader(f"🎫 Ticket Generado")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2", resto[0])
            b3.metric("Bola 3", resto[1])
            b4.metric("Bola 4", resto[2])
            b5.metric("Bola 5", resto[3])

            # --- NUEVA SECCIÓN: GRÁFICOS Y EXPLICACIÓN ---
            st.markdown("### 📊 Análisis del Resultado")
            
            tab_graph, tab_error = st.tabs(["📉 Gráfico de Tendencia", "📋 Tabla de Error"])
            
            with tab_graph:
                # Generar gráfico de dispersión con línea de regresión
                fig, ax = plt.subplots(figsize=(10, 4))
                # Muestreo para no saturar el gráfico
                sample = df.sample(min(500, len(df)))
                ax.scatter(sample['Draw Date'], sample['Num1'], color='blue', alpha=0.3, label='Datos Históricos (Muestra)')
                
                # Línea de predicción (usamos todo el rango de fechas)
                date_range = np.array([X.min(), X.max()]).reshape(-1, 1)
                pred_line = model.predict(date_range)
                ax.plot([df['Draw Date'].min(), df['Draw Date'].max()], pred_line, color='red', linewidth=3, label='Línea de Regresión (IA)')
                
                ax.set_ylabel("Valor del Primer Número")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.markdown(f"""
                <div class="explanation-box">
                <b>💡 Interpretación del Gráfico:</b><br>
                La línea roja representa la "mejor predicción" matemática a través del tiempo. 
                Observe que la línea es casi horizontal (plana). Esto confirma visualmente que <b>no existe una tendencia de subida o bajada</b> 
                en los números. El modelo predice siempre un valor cercano al promedio (aprox. 8-10), lo cual valida la aleatoriedad del juego.
                </div>
                """, unsafe_allow_html=True)

            with tab_error:
                # Comparativa últimos 5 sorteos
                last_5 = df.tail(5).copy()
                last_5['Predicción IA'] = model.predict(last_5[['DrawDate_Ordinal']]).round().astype(int)
                last_5['Error (Diferencia)'] = abs(last_5['Num1'] - last_5['Predicción IA'])
                
                st.write("**Comparativa: Realidad vs IA (Últimos 5 registros)**")
                st.dataframe(last_5[['Draw Date', 'Num1', 'Predicción IA', 'Error (Diferencia)']], use_container_width=True)
                st.caption(f"El R² del modelo es {r2:.5f}, lo que explica el margen de error observado.")

    # === CLASIFICACIÓN CON PROBABILIDADES ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 Clasificación Cash Ball (Árbol de Decisión)")
        
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
            # Obtener probabilidades en lugar de solo predicción
            input_data = [[n1,n2,n3,n4,n5]]
            probs = clf.predict_proba(input_data)[0]
            pred_class = clf.predict(input_data)[0]
            
            st.balloons()
            st.success(f"La Cash Ball más probable es: **{pred_class}**")
            
            # --- NUEVA SECCIÓN: GRÁFICO DE PROBABILIDADES ---
            st.markdown("### 📊 Desglose de Probabilidades")
            
            col_prob, col_desc = st.columns([2, 1])
            
            with col_prob:
                # Crear DataFrame para el gráfico
                prob_df = pd.DataFrame({
                    'Cash Ball': [1, 2, 3, 4],
                    'Probabilidad (%)': probs * 100
                })
                st.bar_chart(prob_df.set_index('Cash Ball'), color="#2196F3")
            
            with col_desc:
                st.markdown(f"""
                <div class="explanation-box">
                <b>💡 ¿Cómo decidió la IA?</b><br>
                El gráfico muestra la confianza del modelo para cada opción.
                <br><br>
                • <b>Probabilidad Cash Ball {pred_class}:</b> {probs[pred_class-1]*100:.1f}%<br>
                • <b>Otras opciones:</b> El resto del porcentaje se distribuye en los otros números.
                <br><br>
                El algoritmo ha analizado combinaciones históricas similares a la ingresada ({n1}, {n2}...) para calcular estos porcentajes.
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")


