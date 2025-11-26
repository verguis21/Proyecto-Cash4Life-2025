import streamlit as st
import pandas as pd
import datetime as dt
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, confusion_matrix

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Cash4Life", layout="wide", page_icon="💰")

# --- 2. ESTILOS CSS (CORREGIDO PARA IPHONE) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FONDO DEGRADADO SUAVE */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        background-attachment: fixed;
    }
    
    /* TARJETAS BLANCAS CON TEXTO OSCURO */
    .block-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid #e0e0e0;
    }
    
    /* FUERZA COLOR DE TEXTO (SOLUCIÓN IOS/DARK MODE) */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #333333 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* TÍTULOS EN VERDE */
    h1 { color: #2e7d32 !important; }
    h2, h3 { color: #388e3c !important; }
    
    /* BOTONES ESTILIZADOS */
    div.stButton > button {
        background: linear-gradient(to right, #43a047, #66bb6a);
        color: white !important;
        border-radius: 10px; border: none;
        padding: 12px 24px; font-size: 16px; font-weight: 600; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
    
    /* CAJAS DE TEXTO EXPLICATIVO */
    .explanation-box { 
        background-color: #f1f8e9; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #8bc34a; 
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin-top: 10px;
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

lottie_robot = load_lottieurl("https://lottie.host/61730045-8c08-4171-8720-c81b37d4566c/2j1y7v3XlQ.json")
lottie_calc = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")

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
    "Ir a:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Semestre:** 2025-II\n**Estado:** Sistema Activo 🟢")

# --- 6. LÓGICA PRINCIPAL ---
if df is not None:
    # Preprocesamiento
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
            st.markdown("### 🎓 Universidad Privada Antenor Orrego")
            st.markdown("---")
            st.markdown("""
            <div style="text-align: justify;">
            Bienvenido al sistema de <b>Aprendizaje Estadístico</b>. Este proyecto analiza miles de sorteos de la lotería 
            Cash4Life (2014-Presente) utilizando algoritmos de <b>Machine Learning</b> para desafiar la aleatoriedad.
            <br><br>
            <b>Objetivo Científico:</b> Determinar si la distribución de los números ganadores sigue un patrón matemático 
            predecible o si obedece estrictamente al azar (distribución uniforme).
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo Oficial:** Mañana, {prox.strftime('%d-%m-%Y')}")

        with c2:
            if lottie_robot: st_lottie(lottie_robot, height=300)

    # === ANÁLISIS ===
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Exploración de Datos")
        
        tab1, tab2, tab3 = st.tabs(["📄 Base de Datos", "📈 Frecuencias", "🔥 Mapa de Correlación"])
        
        with tab1:
            df_vis = df.copy()
            df_vis['Draw Date'] = df_vis['Draw Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_vis[['Draw Date', 'Winning Numbers', 'Cash Ball']], use_container_width=True, height=400)
        
        with tab2:
            st.subheader("🏆 Números Más Frecuentes")
            all_nums = pd.concat([df[f'Num{i}'] for i in range(1, 6)])
            freq = all_nums.value_counts().head(10)
            st.bar_chart(freq, color="#4CAF50")
            
        with tab3:
            st.subheader("🔥 Mapa de Correlación (Heatmap)")
            st.markdown("""
            <div class="explanation-box">
            <b>📘 ¿Qué nos dice este gráfico?</b><br>
            Muestra si existe relación matemática entre los números ganadores. 
            <ul>
            <li><b>Color Claro/Neutro:</b> Indica independencia (Aleatoriedad confirmada).</li>
            <li><b>Color Oscuro/Intenso:</b> Indicaría un patrón sospechoso.</li>
            </ul>
            Este gráfico valida que las bolas salen de forma independiente.
            </div>
            """, unsafe_allow_html=True)
            
            # Matriz de correlación
            corr_cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Cash Ball']
            corr = df[corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f", ax=ax)
            st.pyplot(fig)

    # === PREDICCIÓN (REGRESIÓN) ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia (Regresión)")
        
        # --- MENSAJE IMPORTANTE QUE PEDISTE ---
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ NOTA IMPORTANTE SOBRE EL CÁLCULO:</b><br>
        Este modelo utiliza <b>Inteligencia Artificial (Regresión Lineal)</b> para predecir matemáticamente 
        <b>ÚNICAMENTE LA PRIMERA BOLA (Num1)</b>, basándose en la tendencia histórica de la fecha.<br><br>
        Las bolas restantes (2, 3, 4 y 5) son generadas mediante <b>simulación estocástica (aleatoria)</b> para 
        completar el ticket de juego, ya que estadísticamente dependen del azar puro.
        </div>
        """, unsafe_allow_html=True)
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        with c_input:
            fecha_input = st.date_input("Fecha Objetivo:", dt.date.today() + dt.timedelta(days=1))
            predict_btn = st.button("🚀 Calcular Predicción")
            
        with c_anim:
            anim_placeholder = st.empty()
            
        if predict_btn:
            with c_anim:
                if lottie_calc: st_lottie(lottie_calc, height=150, key="calc")
            with st.spinner("Procesando modelo matemático..."):
                time.sleep(1.5)

            # Predicción IA
            pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = max(1, min(60, int(round(pred_val))))
            
            # Simulación del resto
            resto = np.sort(np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False))
            
            st.markdown("---")
            st.subheader(f"🎫 Ticket Generado")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2 (Random)", resto[0])
            b3.metric("Bola 3 (Random)", resto[1])
            b4.metric("Bola 4 (Random)", resto[2])
            b5.metric("Bola 5 (Random)", resto[3])

            # Gráfico de Tendencia
            st.markdown("### 📉 Análisis de Tendencia")
            fig, ax = plt.subplots(figsize=(10, 3))
            sample = df.sample(min(500, len(df)))
            ax.scatter(sample['Draw Date'], sample['Num1'], color='#90CAF9', alpha=0.5, label='Datos Reales')
            date_range = np.array([X.min(), X.max()]).reshape(-1, 1)
            ax.plot([df['Draw Date'].min(), df['Draw Date'].max()], model.predict(date_range), color='red', linewidth=3, label='Tendencia IA')
            ax.legend()
            st.pyplot(fig)
            st.caption(f"Coeficiente R²: {r2:.5f} (Confirma ausencia de tendencia lineal fuerte).")

    # === CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 Clasificación Cash Ball")
        
        X_class = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y_class = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_class, y_class)
        
        st.write("##### Ingrese la combinación:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Predecir"):
            input_data = [[n1,n2,n3,n4,n5]]
            probs = clf.predict_proba(input_data)[0]
            pred = clf.predict(input_data)[0]
            
            st.success(f"Cash Ball Predicha: **{pred}**")
            
            tab_prob, tab_conf = st.tabs(["📊 Probabilidades", "🧩 Matriz de Confusión"])
            
            with tab_prob:
                prob_df = pd.DataFrame({'Opción': [1,2,3,4], 'Probabilidad': probs})
                st.bar_chart(prob_df.set_index('Opción'), color="#2196F3")
                
            with tab_conf:
                st.markdown("""
                <div class="explanation-box">
                <b>📘 Matriz de Confusión:</b><br>
                Esta tabla compara las predicciones del modelo contra la realidad histórica. 
                Permite ver dónde se "equivoca" más la IA (Diagonal principal = Aciertos).
                </div>
                """, unsafe_allow_html=True)
                
                # Matriz de Confusión real
                y_pred_all = clf.predict(X_class)
                cm = confusion_matrix(y_class, y_pred_all)
                
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicción IA')
                ax_cm.set_ylabel('Valor Real')
                st.pyplot(fig_cm)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")


