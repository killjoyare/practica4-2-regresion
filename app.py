@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')

    con.execute("SELECT setseed(0.42);")

    # Generar primero las variables independientes
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

    # Calcular el consumo con las variables guardadas
    con.execute("""
    CREATE TABLE telemetria_agua AS
    SELECT
        residencia_id,
        temperatura_max,
        humedad_max,
        personas_familia,
        consumo_bidon_pc,
        ROUND(
            94.0 + (2.5 * temperatura_max)
            - (0.8 * humedad_max)
            - (5.2 * personas_familia)
            + (15.0 * consumo_bidon_pc)
            + (random() * 6.0 - 3.0), 2
        ) AS cpc_consumo
    FROM variables_agua;
    """)

    return con.execute(
        "SELECT * FROM telemetria_agua"
    ).df()
