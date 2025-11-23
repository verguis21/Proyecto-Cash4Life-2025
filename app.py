import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# Configuración de página
st.set_page_config(page_title="Predicción Cash4Life", layout="wide")

st.title("💰 Sistema de Predicción Cash4Life - New York")
st.markdown("""
**Autores:** Bernabé Arce, Coronado Medina, Enriquez Cabanillas, et al.
**Curso:** Aprendizaje Estadístico
""")

# Carga de datos
@st.cache_data
def load_data():
    file_path = "Lottery_Cash_4_Life_Winning_Numbers__Beginning_2014.csv"
    try:
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Preprocesamiento básico
    df['DrawDate_Ordinal'] = df['Draw Date'].map(dt.datetime.toordinal)
    # Intentar separar números si vienen juntos
    try:
        nums = df["Winning Numbers"].str.split(" ", expand=True)
        for i in range(5):
            df[f'Num{i+1}'] = pd.to_numeric(nums[i])
    except:
        pass

    # Menú lateral
    menu = st.sidebar.selectbox("Seleccione Módulo", ["Análisis", "Predicción (Regresión)", "Clasificación (Cash Ball)"])

    if menu == "Análisis":
        st.subheader("📊 Datos Históricos")
        st.dataframe(df.head())
        st.write(df.describe())

    elif menu == "Predicción (Regresión)":
        st.subheader("📈 Predicción del Primer Número")
        X = df[['DrawDate_Ordinal']]
        y = df['Num1']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        st.metric("Precisión R²", f"{r2:.4f}")
        
        fecha = st.date_input("Seleccione fecha futura")
        if st.button("Predecir"):
            pred = model.predict([[dt.datetime.toordinal(fecha)]])
            st.success(f"Predicción Num1: {pred[0]:.2f}")

    elif menu == "Clasificación (Cash Ball)":
        st.subheader("🟢 Predicción Cash Ball")
        # Usamos los 5 números para predecir la bola extra
        X = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]
        y = df['Cash Ball']
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        n1 = c1.number_input("N1", 1, 60, 5)
        n2 = c2.number_input("N2", 1, 60, 10)
        n3 = c3.number_input("N3", 1, 60, 25)
        n4 = c4.number_input("N4", 1, 60, 30)
        n5 = c5.number_input("N5", 1, 60, 45)
        
        if st.button("Calcular Cash Ball"):
            pred = model.predict([[n1,n2,n3,n4,n5]])
            st.info(f"Cash Ball Esperada: {pred[0]}")

else:
    st.error("Error: No se encuentra el archivo CSV. Súbelo a GitHub.")