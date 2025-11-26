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

# --- 2. ESTILOS CSS (PROFESIONAL) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FONDO */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        background-attachment: fixed;
    }
    
    /* CONTENEDOR */
    .block-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid #e0e0e0;
    }
    
    /* TEXTO OSCURO PARA TODOS (FIX IPHONE) */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #333333 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    h1 { color: #2e7d32 !important; }
    h2, h3 { color: #388e3c !important; }
    
    /* BOTONES */
    div.stButton > button {
        background: linear-gradient(to right, #43a047, #66bb6a);
        color: white !important;
        border-radius: 10px; padding: 12px 24px; font-weight: 600; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
    
    /* CAJAS DE EXPLICACIÓN (ESTILO TESIS) */
    .explanation-box { 
        background-color: #f1f8e9; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #8bc34a; 
        margin-top: 15px; margin-bottom: 15px;
        font-size: 15px;
    }
    .science-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin-top: 10px;
        font-size: 15px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin-bottom: 20px;
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

# --- 6. APP ---
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
            st.markdown("### 🎓 Universidad Privada Antenor Orrego")
            st.markdown("---")
            st.markdown("""
            <div style="text-align: justify;">
            <b>Investigación:</b> Análisis de patrones estocásticos en loterías mediante Machine Learning.
            <br><br>
            Este sistema no es una herramienta de juego, sino un instrumento científico para validar la <b>Hipótesis Nula de Aleatoriedad</b>. 
            Procesamos miles de sorteos históricos para determinar si existen sesgos matemáticos explotables.
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
        
        tab1, tab2, tab3 = st.tabs(["📄 Datos", "📈 Frecuencias", "🔥 Mapa de Correlación"])
        
        with tab1:
            df_vis = df.copy()
            df_vis['Draw Date'] = df_vis['Draw Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_vis[['Draw Date', 'Winning Numbers', 'Cash Ball']], use_container_width=True, height=400)
        
        with tab2:
            st.subheader("🏆 Números Más Frecuentes")
            st.markdown("""
            <div class="explanation-box">
            <b>📘 Interpretación:</b> En una aleatoriedad perfecta, todas las barras serían iguales. 
            Las diferencias de altura muestran la varianza natural de la muestra histórica ("Números Calientes").
            </div>
            """, unsafe_allow_html=True)
            all_nums = pd.concat([df[f'Num{i}'] for i in range(1, 6)])
            freq = all_nums.value_counts().head(10)
            st.bar_chart(freq, color="#4CAF50")
            
        with tab3:
            st.subheader("🔥 Mapa de Correlación (Heatmap)")
            st.markdown("""
            <div class="science-box">
            <b>🧪 Análisis de Independencia:</b><br>
            • <b>Colores Claros:</b> Indican correlación cero (Independencia total).<br>
            • <b>Colores Oscuros:</b> Indicarían dependencia (Patrón sospechoso).<br>
            <b>Conclusión:</b> El predominio de colores claros valida científicamente que las bolas no se influyen entre sí.
            </div>
            """, unsafe_allow_html=True)
            
            corr = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Cash Ball']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f", ax=ax)
            st.pyplot(fig)

    # === PREDICCIÓN ===
    elif menu == "🔮 Predicción (Regresión)":
        st.header("🔮 Predicción de Tendencia")
        
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ NOTA TÉCNICA:</b> La IA utiliza <b>Regresión Lineal</b> para predecir la tendencia de la primera bola. 
        Las bolas restantes (2-5) se generan por simulación estocástica para completar el ticket.
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
            
        if predict_btn:
            with c_anim:
                if lottie_calc: st_lottie(lottie_calc, height=150, key="calc")
            with st.spinner("Procesando..."):
                time.sleep(1.5)

            # Predicción con explicación de decimales
            pred_val_float = model.predict([[dt.datetime.toordinal(fecha_input)]])[0]
            n1 = max(1, min(60, int(round(pred_val_float))))
            resto = np.sort(np.random.choice(list(set(range(1, 61)) - {n1}), 4, replace=False))
            
            st.markdown("---")
            st.subheader(f"🎫 Resultado del Modelo")
            
            # Explicación del redondeo
            st.markdown(f"""
            <div class="science-box">
            <b>🧮 Cálculo Matemático vs. Realidad:</b><br>
            La Regresión calculó el valor exacto de tendencia: <b>{pred_val_float:.4f}</b>.<br>
            Como la lotería es discreta, el sistema lo interpreta como la bola: <b>{n1}</b>.
            </div>
            """, unsafe_allow_html=True)

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Bola 1 (IA)", n1)
            b2.metric("Bola 2", resto[0])
            b3.metric("Bola 3", resto[1])
            b4.metric("Bola 4", resto[2])
            b5.metric("Bola 5", resto[3])

            st.markdown("### 📊 Justificación Gráfica")
            tab_graph, tab_error = st.tabs(["📉 Línea de Tendencia", "📋 Análisis de Error"])
            
            with tab_graph:
                fig, ax = plt.subplots(figsize=(10, 3))
                sample = df.sample(min(500, len(df)))
                ax.scatter(sample['Draw Date'], sample['Num1'], color='#90CAF9', alpha=0.5, label='Historial Real')
                date_range = np.array([X.min(), X.max()]).reshape(-1, 1)
                ax.plot([df['Draw Date'].min(), df['Draw Date'].max()], model.predict(date_range), color='red', linewidth=3, label='Predicción IA')
                ax.legend()
                st.pyplot(fig)
                
                st.markdown("""
                <div class="explanation-box">
                <b>💡 Interpretación:</b> La línea roja es casi <b>plana</b>. Esto demuestra visualmente que el paso del tiempo 
                no afecta el resultado. La mejor predicción matemática es el promedio histórico.
                </div>
                """, unsafe_allow_html=True)

            with tab_error:
                last_5 = df.tail(5).copy()
                last_5['Draw Date'] = last_5['Draw Date'].dt.strftime('%Y-%m-%d')
                last_5['Predicción'] = model.predict(last_5[['DrawDate_Ordinal']]).round().astype(int)
                last_5['Diferencia'] = abs(last_5['Num1'] - last_5['Predicción'])
                st.dataframe(last_5[['Draw Date', 'Num1', 'Predicción', 'Diferencia']], use_container_width=True)
                
                st.markdown("""
                <div class="science-box">
                <b>🧪 ¿Por qué hay error?</b> Un error alto aquí es <b>bueno científicamente</b>. 
                Confirma que los números reales saltan aleatoriamente lejos de la predicción promedio, 
                validando que el sorteo no está manipulado.
                </div>
                """, unsafe_allow_html=True)

    # === CLASIFICACIÓN ===
    elif menu == "🟢 Clasificación (Cash Ball)":
        st.header("🟢 Clasificación Cash Ball")
        
        X_class = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y_class = df['Cash Ball']
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_class, y_class)
        
        st.write("##### Combinación de entrada:")
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("B1", 1, 60, 5)
        n2 = c2.number_input("B2", 1, 60, 10)
        n3 = c3.number_input("B3", 1, 60, 25)
        n4 = c4.number_input("B4", 1, 60, 30)
        n5 = c5.number_input("B5", 1, 60, 45)
        
        if st.button("🎱 Analizar Patrón"):
            probs = clf.predict_proba([[n1,n2,n3,n4,n5]])[0]
            pred = clf.predict([[n1,n2,n3,n4,n5]])[0]
            
            st.success(f"Cash Ball Predicha: **{pred}**")
            
            tab_prob, tab_conf = st.tabs(["📊 Probabilidades", "🧩 Matriz de Confusión"])
            
            with tab_prob:
                prob_df = pd.DataFrame({'Bola': [1,2,3,4], 'Confianza': probs})
                st.bar_chart(prob_df.set_index('Bola'), color="#2196F3")
                st.markdown("""
                <div class="explanation-box">
                El gráfico muestra la <b>incertidumbre del modelo</b>. La IA no adivina, sino que asigna porcentajes 
                de probabilidad a cada opción basándose en patrones pasados.
                </div>
                """, unsafe_allow_html=True)
                
            with tab_conf:
                y_pred_all = clf.predict(X_class)
                cm = confusion_matrix(y_class, y_pred_all)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicción IA')
                ax_cm.set_ylabel('Valor Real')
                st.pyplot(fig_cm)
                
                st.markdown("""
                <div class="science-box">
                <b>📘 Interpretación:</b> Esta matriz compara Aciertos vs. Errores. La dispersión fuera de la diagonal principal 
                demuestra la dificultad inherente de clasificar eventos puramente aleatorios.
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("⚠️ Error: No se encontró el dataset en GitHub.")
