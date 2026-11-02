import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")

st.title("📊 Práctica 4.2: Regresión Lineal y Correlación")
st.subheader("Caso de Estudio: Predicción del Consumo Per Cápita de Agua Potable (CPC)")

@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    con.execute("""
    CREATE TABLE telemetria_agua AS
    SELECT
        residencia_id,
        ROUND(18 + (random() * 14), 2) AS temperatura_max,
        ROUND(50 + (random() * 40), 2) AS humedad_max,
        ROUND(3 + (random() * 4), 0) AS personas_familia,
        ROUND(0.2 + (random() * 1.8), 2) AS consumo_bidon_pc,
        ROUND(
            94.0 + (2.5 * (18 + random() * 14))
                 - (0.8 * (50 + random() * 40))
                 - (5.2 * (3 + random() * 4))
                 + (15.0 * (0.2 + random() * 1.8))
                 + (random() * 6.0 - 3.0), 2
        ) AS cpc_consumo
    FROM generate_series(1, 100) AS t(residencia_id);
    """)
    return con.execute("SELECT * FROM telemetria_agua").df()

df_agua = cargar_datos()

tab1, tab2, tab3 = st.tabs(["📋 Datos y Correlación", "📈 Modelo OLS y Métricas", "📊 Panel de Diagnóstico Visual"])

with tab1:
    st.write("### --- PASO 1: PRIMEROS 5 REGISTROS OBTENIDOS DE DUCKDB ---")
    st.markdown(df_agua.head().to_markdown(index=False))

    matriz_correlacion = df_agua.drop(columns=['residencia_id']).corr(method='pearson')
    
    st.write("### --- PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r) ---")
    st.markdown(matriz_correlacion.round(4).to_markdown())

X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

with tab2:
    st.write("### 📉 Paso 3: Resumen del Modelo OLS")
    # Mostramos de forma limpia el R cuadrado tal como lo solicitaste
    r2_val = modelo_ols.rsquared
    st.markdown(f"**Coeficiente de Determinación ($R^2$):** {r2_val * 100:.2f}% ({r2_val:.4f})")

residuos = modelo_ols.resid
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)

model_sk = LinearRegression()
model_sk.fit(X, Y)
y_pred = model_sk.predict(X)

r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))

with tab2:
    st.markdown("---")
    st.write("### --- PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS ---")
    st.write(f"1. **Prueba de Normalidad (Shapiro-Wilk):** Estadístico = {stat_sw:.4f}, valor p = {p_val_sw:.4f}")
    if p_val_sw > 0.05:
        st.success("   -> CONCLUSIÓN: Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).")
    else:
        st.warning("   -> CONCLUSIÓN: Se rechaza la normalidad de los residuos (p < 0.05).")
    
    st.write(f"2. **Estadístico Durbin-Watson (Independencia):** {dw_stat:.4f} (Valores entre 1.5 y 2.5 indican ausencia de autocorrelación.)")

    st.markdown("---")
    st.write("### --- PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn) ---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Coeficiente de Determinación (R²)", value=f"{r2_sk * 100:.2f}%")
    with col2:
        st.metric(label="Error Cuadrático Medio Raíz (RMSE)", value=f"{rmse_sk:.4f} lt/hab-día")

    ecuacion_txt = f"CPC = {model_sk.intercept_:.2f} + ({model_sk.coef_[0]:.2f} * Temp) + ({model_sk.coef_[1]:.2f} * Humedad) + ({model_sk.coef_[2]:.2f} * Personas) + ({model_sk.coef_[3]:.2f} * Bidon)"
    st.info(f"**Ecuación del Modelo Ajustado:**\n\n{ecuacion_txt}")

with tab3:
    st.write("### --- PANEL DE DIAGNÓSTICO VISUAL ---")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=16, fontweight='bold')

    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
    axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

    sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1],
                line_kws={'color': 'red', 'linewidth': 2})
    axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
    axes[0, 1].set_xlabel('Humedad Máxima (%)')
    axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

    axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
    axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
    axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_sk*100:.1f}%)')
    axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
    axes[0, 2].set_ylabel('CPC Real (y)')

    axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
    axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
    axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

    sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
    axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

    axes[1, 2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
