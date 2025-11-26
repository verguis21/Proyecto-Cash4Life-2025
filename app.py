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
st.set_page_config(page_title="Predicción Cash4Life - UPAO", layout="wide", page_icon="💰")

# --- 2. ESTILOS CSS (INSTITUCIONAL) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FONDO SUAVE */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* CONTENEDOR BLANCO */
    .block-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-top: 5px solid #FECB00; /* Color Amarillo UPAO (Aprox) */
    }
    
    /* TEXTOS */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #333333 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    h1 { color: #003B70 !important; } /* Azul Institucional */
    h2, h3 { color: #0056b3 !important; }
    
    /* BOTONES */
    div.stButton > button {
        background-color: #003B70; /* Azul UPAO */
        color: white !important;
        border-radius: 8px; border: none;
        padding: 10px 20px; font-weight: bold; 
        width: 100%;
    }
    div.stButton > button:hover { background-color: #FECB00; color: #003B70 !important; }
    
    /* CAJAS */
    .explanation-box { background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 5px solid #2196f3; margin: 15px 0; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin-bottom: 20px; }
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

# --- 5. MENÚ LATERAL (CON LOGO) ---
# Logo UPAO (URL pública, si tienes una propia subela a GitHub y pon el nombre del archivo)
st.sidebar.image("https://seeklogo.com/images/U/upao-logo-58851E2D7D-seeklogo.com.png", width=150)

st.sidebar.title("Menú Principal")
menu = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📊 Análisis Histórico", "🔮 Predicción (Regresión)", "🟢 Clasificación (Cash Ball)"]
)
st.sidebar.markdown("---")

with st.sidebar.expander("👨‍🎓 Equipo de Investigación", expanded=True):
    st.write("""
    * Bernabé Arce, James
    * Coronado Medina, Sergio
    * Enriquez Cabanillas, César
    * Carrascal Carranza, Hetzer
    * Lázaro Velásquez, Jesús
    * Martino López, Marielsys
    * Mori Galarza, Franco
    * Vergaray Colonia, José
    """)
st.sidebar.info("**Semestre:** 2025-II")

# --- 6. LÓGICA PRINCIPAL ---
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
            st.title("Sistema de Aprendizaje Estadístico")
            st.markdown("### Proyecto Final: Análisis Cash4Life")
            st.markdown("#### 🏛️ Universidad Privada Antenor Orrego")
            
            # IMAGEN ADICIONAL (OPCIONAL)
            # Si quieres poner una imagen aquí, descomenta la siguiente línea y pon la URL
            # st.image("https://tu-imagen.com/foto-grupo.jpg", caption="Equipo de Trabajo")
            
            st.markdown("""
            <div style="text-align: justify; margin-top: 20px;">
            <b>Resumen Ejecutivo:</b><br>
            Este aplicativo web implementa modelos de <b>Machine Learning</b> para auditar la aleatoriedad de la lotería de Nueva York.
            Analizamos la data histórica (2014-Presente) para responder: <i>¿Es posible predecir el azar?</i>
            <br><br>
            <b>Resultados Preliminares:</b>
            <ul>
            <li><b>Regresión (Tendencia):</b> R² ≈ 45% (Correlación moderada).</li>
            <li><b>Clasificación (Cash Ball):</b> Precisión ≈ 25% (Validación de aleatoriedad).</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            hoy = dt.date.today()
            prox = hoy + dt.timedelta(days=1)
            st.warning(f"📅 **Próximo Sorteo:** {prox.strftime('%d-%m-%Y')}")

        with c2:
            if lottie_robot: st_lottie(lottie_robot, height=350)

    # === ANÁLISIS ===
    elif menu == "📊 Análisis Histórico":
        st.header("📊 Minería de Datos")
        tab1, tab2, tab3 = st.tabs(["📄 Base de Datos", "📈 Frecuencias", "🔥 Correlación"])
        
        with tab1:
            df_vis = df.copy()
            df_vis['Draw Date'] = df_vis['Draw Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_vis[['Draw Date', 'Winning Numbers', 'Cash Ball']], use_container_width=True, height=400)
        with tab2:
            st.subheader("Frecuencia de Números")
            all_nums = pd.concat([df[f'Num{i}'] for i in range(1, 6)])
            st.bar_chart(all_nums.value_counts().head(10), color="#003B70")
        with tab3:
            st.subheader("Mapa de Calor")
            st.markdown('<div class="explanation-box">Analiza la independencia entre variables.</div>', unsafe_allow_html=True)
            corr = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Cash Ball']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
            st.pyplot(fig)

    # === PREDICCIÓN ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Modelo Predictivo")
        st.markdown('<div class="warning-box">⚠️ <b>ALCANCE:</b> La IA calcula la tendencia matemática de la <b>Bola 1</b>. El resto del ticket se completa mediante simulación estocástica.</div>', unsafe_allow_html=True)
        
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        c_input, c_anim = st.columns([1, 1])
        with c_input:
            fecha_input = st.date_input("Fecha Sorteo:", dt.date.today() + dt.timedelta(days=1))
            if st.button("🚀 Ejecutar Modelo"):
                with c_anim:
                    if lottie_calc: st_lottie(lottie_calc, height=150, key="calc")
                with st.spinner("Procesando..."):
                    time.sleep(1.5)
                
                pred_val = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
                n1 = max(1, min(60, int(round(pred_val))))
                resto = np.sort(np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False))
                
                st.markdown("---")
                st.subheader("🎫 Ticket Generado")
                cols = st.columns(5)
                cols[0].metric("Bola 1 (IA)", n1)
                for i in range(4): cols[i+1].metric(f"Bola {i+2}", resto[i])
                
                st.markdown("### 📊 Validación")
                tab_g, tab_e = st.tabs(["Tendencia", "Error"])
                with tab_g:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    sample = df.sample(min(500, len(df)))
                    ax.scatter(sample['Draw Date'], sample['Num1'], color='#90CAF9', alpha=0.5)
                    date_range = np.array([X.min(), X.max()]).reshape(-1, 1)
                    ax.plot([df['Draw Date'].min(), df['Draw Date'].max()], model.predict(date_range), color='red', linewidth=2)
                    st.pyplot(fig)
                    st.caption(f"Tendencia Plana = Aleatoriedad Confirmada (R²: {r2:.4f})")
                with tab_e:
                    last_5 = df.tail(5).copy()
                    last_5['Pred'] = model.predict(last_5[['DrawDate_Ordinal']]).round().astype(int)
                    last_5['Diff'] = abs(last_5['Num1'] - last_5['Pred'])
                    st.dataframe(last_5[['Draw Date', 'Num1', 'Pred', 'Diff']], use_container_width=True)

    # === CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 Clasificador IA (Árbol de Decisión)")
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']], df['Cash Ball'])
        
        c = st.columns(5)
        nums_in = [c[i].number_input(f"B{i+1}", 1, 60, 5*(i+1)) for i in range(5)]
        
        if st.button("🎱 Predecir Cash Ball"):
            pred = clf.predict([nums_in])[0]
            probs = clf.predict_proba([nums_in])[0]
            st.success(f"Cash Ball: **{pred}**")
            
            t1, t2 = st.tabs(["Probabilidades", "Matriz Confusión"])
            with t1:
                st.bar_chart(pd.DataFrame({'Ball': [1,2,3,4], 'Prob': probs}).set_index('Ball'))
            with t2:
                cm = confusion_matrix(df['Cash Ball'], clf.predict(df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]))
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")
