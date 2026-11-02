
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

st.set_page_config(
    page_title="Práctica 4.2 - Regresión Lineal",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Práctica 4.2: Regresión Lineal y Correlación")
st.subheader(
    "Caso de Estudio: Predicción del Consumo Per Cápita de Agua Potable (CPC)"
)

# Configuración para visualizar correctamente los datos
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.expand_frame_repr', False)


# --------------------------------------------------
# PASO 1: GENERACIÓN DE DATOS CON DUCKDB
# --------------------------------------------------

@st.cache_data
def cargar_datos():

    con = duckdb.connect(database=':memory:')

    con.execute("SELECT setseed(0.42);")

    # Primero generamos las variables independientes
    con.execute("""
    CREATE TEMP TABLE variables_agua AS
    SELECT
        residencia_id,
        ROUND(18 + random() * 14, 2) AS temperatura_max,
        ROUND(50 + random() * 40, 2) AS humedad_max,
        ROUND(3 + random() * 4, 0) AS personas_familia,
        ROUND(0.2 + random() * 1.8, 2) AS consumo_bidon_pc
    FROM generate_series(1, 100) AS t(residencia_id);
    """)

    # Después calculamos el consumo de agua
    con.execute("""
    CREATE TABLE telemetria_agua AS
    SELECT
        residencia_id,
        temperatura_max,
        humedad_max,
        personas_familia,
        consumo_bidon_pc,
        ROUND(
            ROUND(
    94.0
    + (2.5 * temperatura_max)
    - (0.8 * humedad_max)
    - (5.2 * personas_familia)
    + (15.0 * consumo_bidon_pc)
    + (random() * 6.0 - 3.0),
    2
) AS cpc_consumo
    FROM variables_agua;
    """)

    df = con.execute(
        "SELECT * FROM telemetria_agua"
    ).df()

    con.close()

    return df


df_agua = cargar_datos()


# --------------------------------------------------
# PASO 2: MATRIZ DE CORRELACIÓN
# --------------------------------------------------

matriz_correlacion = df_agua.drop(
    columns=['residencia_id']
).corr(method='pearson')


# --------------------------------------------------
# PASO 3: MODELO DE REGRESIÓN LINEAL MÚLTIPLE OLS
# --------------------------------------------------

X = df_agua[
    [
        'temperatura_max',
        'humedad_max',
        'personas_familia',
        'consumo_bidon_pc'
    ]
]

Y = df_agua['cpc_consumo']

X_const = sm.add_constant(X)

modelo_ols = sm.OLS(Y, X_const).fit()


# --------------------------------------------------
# PASO 4: DIAGNÓSTICO DE RESIDUOS
# --------------------------------------------------

residuos = modelo_ols.resid

stat_sw, p_val_sw = stats.shapiro(residuos)

dw_stat = durbin_watson(residuos)


# --------------------------------------------------
# PASO 5: MODELO CON SCIKIT-LEARN
# --------------------------------------------------

model_sk = LinearRegression()

model_sk.fit(X, Y)

y_pred = model_sk.predict(X)

r2_sk = r2_score(Y, y_pred)

rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))


# --------------------------------------------------
# ORGANIZACIÓN DE LA APLICACIÓN
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Datos y Correlación",
    "📈 Modelo OLS y Métricas",
    "📉 Panel de Diagnóstico Visual"
])


# ==================================================
# PESTAÑA 1: DATOS Y CORRELACIÓN
# ==================================================

with tab1:

    st.write("### PASO 1: PRIMEROS 5 REGISTROS OBTENIDOS DE DUCKDB")

    st.dataframe(
        df_agua.head(),
        use_container_width=True,
        hide_index=True
    )

    st.write("### PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r)")

    st.dataframe(
        matriz_correlacion.round(4),
        use_container_width=True
    )

    st.write("### Correlación de las variables con el consumo de agua")

    correlaciones = matriz_correlacion['cpc_consumo'].drop(
        'cpc_consumo'
    ).sort_values(ascending=False)

    st.dataframe(
        correlaciones.to_frame("Correlación con CPC"),
        use_container_width=True
    )

    st.write("### Resumen estadístico de los datos")

    st.dataframe(
        df_agua.describe().round(2),
        use_container_width=True
    )


# ==================================================
# PESTAÑA 2: MODELO OLS Y MÉTRICAS
# ==================================================

with tab2:

    st.write("### PASO 3: RESUMEN DEL MODELO OLS")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R cuadrado (R²)",
            f"{modelo_ols.rsquared * 100:.2f}%"
        )

    with col2:
        st.metric(
            "R cuadrado ajustado",
            f"{modelo_ols.rsquared_adj * 100:.2f}%"
        )

    with st.expander("Ver resumen estadístico completo del modelo OLS"):
        st.code(
            modelo_ols.summary().as_text(),
            language="text"
        )

    st.markdown("---")

    st.write("### PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS")

    st.write(
        f"**Prueba de normalidad (Shapiro-Wilk):** "
        f"Estadístico = {stat_sw:.4f}, valor p = {p_val_sw:.4f}"
    )

    if p_val_sw > 0.05:
        st.success(
            "CONCLUSIÓN: Los residuos cumplen con el supuesto "
            "de normalidad (p > 0.05)."
        )
    else:
        st.warning(
            "CONCLUSIÓN: Se rechaza la normalidad de los residuos (p < 0.05)."
        )

    st.write(
        f"**Estadístico Durbin-Watson:** {dw_stat:.4f}"
    )

    if 1.5 <= dw_stat <= 2.5:
        st.success(
            "CONCLUSIÓN: No se observa autocorrelación importante "
            "en los residuos."
        )
    else:
        st.warning(
            "CONCLUSIÓN: Se recomienda revisar la independencia de los residuos."
        )

    st.markdown("---")

    st.write("### PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn)")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Coeficiente de Determinación (R²)",
            value=f"{r2_sk * 100:.2f}%"
        )

    with col2:
        st.metric(
            label="Error Cuadrático Medio Raíz (RMSE)",
            value=f"{rmse_sk:.4f} lt/hab-día"
        )

    ecuacion_txt = (
        f"CPC = {model_sk.intercept_:.2f} "
        f"+ ({model_sk.coef_[0]:.2f} × Temperatura) "
        f"+ ({model_sk.coef_[1]:.2f} × Humedad) "
        f"+ ({model_sk.coef_[2]:.2f} × Personas) "
        f"+ ({model_sk.coef_[3]:.2f} × Bidón)"
    )

    st.info(
        f"**Ecuación del modelo ajustado:**\n\n{ecuacion_txt}"
    )


# ==================================================
# PESTAÑA 3: PANEL DE DIAGNÓSTICO VISUAL
# ==================================================

with tab3:

    st.write("### PANEL DE DIAGNÓSTICO VISUAL")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(
        'Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)',
        fontsize=16,
        fontweight='bold'
    )

    # Gráfica 1: Matriz de correlación
    sns.heatmap(
        matriz_correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title(
        '1. Matriz de Correlación de Pearson (r)'
    )

    # Gráfica 2: Humedad vs consumo
    sns.regplot(
        data=df_agua,
        x='humedad_max',
        y='cpc_consumo',
        color='b',
        ax=axes[0, 1],
        line_kws={'color': 'red', 'linewidth': 2}
    )

    axes[0, 1].set_title(
        '2. Tendencia: Humedad Máx vs Consumo'
    )

    axes[0, 1].set_xlabel('Humedad Máxima (%)')
    axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

    # Gráfica 3: Valores reales vs predichos
    axes[0, 2].scatter(
        y_pred,
        Y,
        alpha=0.7,
        color='teal'
    )

    axes[0, 2].plot(
        [Y.min(), Y.max()],
        [Y.min(), Y.max()],
        'r--',
        lw=2
    )

    axes[0, 2].set_title(
        f'3. Valores Reales vs Predichos (R² = {r2_sk * 100:.1f}%)'
    )

    axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
    axes[0, 2].set_ylabel('CPC Real (y)')

    # Gráfica 4: Residuos vs ajustados
    axes[1, 0].scatter(
        y_pred,
        residuos,
        color='purple',
        alpha=0.6
    )

    axes[1, 0].axhline(
        y=0,
        color='r',
        linestyle='--',
        lw=2
    )

    axes[1, 0].set_title(
        '4. Residuos vs Ajustados (Homocedasticidad)'
    )

    axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
    axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

    # Gráfica 5: Q-Q Normal
    sm.qqplot(
        residuos,
        line='45',
        fit=True,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title(
        '5. Gráfico Q-Q Normal (Residuos)'
    )

    # Ocultar el sexto espacio
    axes[1, 2].axis('off')

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Práctica 4.2 | Regresión Lineal Múltiple | "
    "DuckDB, Pandas, Statsmodels, SciPy y Scikit-learn"
)
