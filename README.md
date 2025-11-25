#  Análisis Estadístico y Predicción: Lotería Cash4Life

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Model-Scikit_Learn-orange)

##  Descripción del Proyecto
Este proyecto desarrolla un análisis estadístico profundo sobre los resultados históricos de la lotería **Cash4Life (New York)** desde el año 2014. A pesar de que los sorteos se diseñan bajo principios de aleatoriedad, esta investigación busca identificar patrones, sesgos o tendencias utilizando técnicas de **Machine Learning**.

El sistema permite visualizar datos, realizar predicciones de regresión lineal y clasificar resultados probables mediante árboles de decisión.

##  Autores
**Universidad Privada Antenor Orrego - Ingeniería de Sistemas e IA**
* Bernabé Arce, James Franco
* Coronado Medina, Sergio Adrian
* Enriquez Cabanillas, César
* Carrascal Carranza, Hetzer
* Lázaro Velásquez, Jesús Alberto
* Martino López, Marielsys Paola
* Mori Galarza, Franco
* Vergaray Colonia, José Francisco

## Funcionalidades del Sistema
El aplicativo web cuenta con tres módulos principales:

### 1. Análisis Exploratorio de Datos
Visualización de la data cruda y estadísticas descriptivas de los sorteos históricos para validar la integridad de la información.

### 2. Predicción de Tendencia (Regresión)
Utiliza un modelo de **Regresión Lineal Simple** para analizar la relación entre el paso del tiempo y el valor del primer número ganador (*Num1*). 
* **Objetivo:** Determinar si existe una tendencia predecible ascendente o descendente en los sorteos.
* **Métrica:** Coeficiente de determinación ($R^2$).

### 3. Clasificación (Cash Ball)
Implementa un algoritmo de **Árbol de Decisión** para predecir el número especial (*Cash Ball*) basándose en los 5 números principales sorteados.

##  Tecnologías Usadas
* **Lenguaje:** Python
* **Interfaz Web:** Streamlit
* **Ciencia de Datos:** Pandas, Scikit-Learn, Numpy
* **Visualización:** Matplotlib (en notebooks), Lottie Files (animaciones)

## 📄 Instalación Local
Si deseas correr este proyecto en tu computadora:

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/verguis21/Proyecto-Cash4Life-2025] (https://github.com/verguis21/Proyecto-Cash4Life-2025)
