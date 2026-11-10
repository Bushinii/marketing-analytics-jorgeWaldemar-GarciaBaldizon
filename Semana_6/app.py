# Archivo base para el despliegue del Agente en Streamlit

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

# Streamlit permite crear la aplicación web interactiva.
import streamlit as st

# Pandas permite leer, manipular y analizar datos tabulares.
import pandas as pd

# NumPy permite realizar operaciones matemáticas y trabajar con arreglos.
import numpy as np

# Plotly Express permite generar gráficos interactivos.
import plotly.express as px

# Herramientas para preparar los datos antes de aplicar los algoritmos.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Algoritmos de agrupamiento o clustering.
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Métricas para evaluar y comparar los resultados de los modelos.
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Implementación del algoritmo K-Prototypes.
from kmodes.kprototypes import KPrototypes


# ============================================================
# 2. CONFIGURACIÓN GENERAL DE STREAMLIT
# ============================================================

# Define el título de la pestaña, el ícono y el diseño de la aplicación.
st.set_page_config(
    page_title="Segmentación de clientes",
    page_icon="🛒",
    layout="wide"
)

# Título principal que aparece en la aplicación.
st.title("🛒 Laboratorio de segmentación de clientes")

# Texto descriptivo debajo del título.
st.caption(
    "Explora perfiles de clientes y compara cinco técnicas de segmentación."
)


# ============================================================
# 3. FUNCIÓN PARA GENERAR EL PERFIL DE LOS SEGMENTOS
# ============================================================

def profile_table(df, group_col, spend_col):

    # Creamos una copia para evitar modificar el DataFrame original.
    x = df.copy()

    # Convertimos los identificadores de grupo a texto.
    # Esto permite manejar etiquetas numéricas y categóricas.
    x[group_col] = x[group_col].astype(str)

    # Calculamos el número total de clientes.
    total_n = len(x)

    # Calculamos el gasto total de todos los clientes.
    # Si no se seleccionó una variable de gasto, utilizamos NaN.
    total_spend = (
        pd.to_numeric(x[spend_col], errors="coerce").sum()
        if spend_col else np.nan
    )

    # Lista donde almacenaremos los resultados de cada segmento.
    rows = []

    # Recorremos cada grupo y obtenemos los clientes que pertenecen a él.
    for group, g in x.groupby(group_col, dropna=False):

        # Convertimos la variable de gasto a valores numéricos.
        spend = (
            pd.to_numeric(g[spend_col], errors="coerce")
            if spend_col else pd.Series(dtype=float)
        )

        # Sumamos el gasto de todos los clientes del segmento.
        group_sum = spend.sum() if spend_col else np.nan

        # Agregamos las métricas calculadas a la lista.
        rows.append({

            # Nombre o identificador del segmento.
            "Segmento": group,

            # Cantidad de clientes pertenecientes al segmento.
            "Clientes": len(g),

            # Porcentaje de clientes que representa el segmento.
            "% de clientes": round(100 * len(g) / total_n, 2),

            # Gasto promedio de los clientes del segmento.
            "Gasto promedio (US$)": (
                round(spend.mean(), 2) if spend_col else np.nan
            ),

            # Porcentaje del gasto total que representa el segmento.
            "% del gasto": (
                round(100 * group_sum / total_spend, 2)
                if spend_col and total_spend else np.nan
            )
        })

    # Convertimos los resultados en una tabla de Pandas.
    return pd.DataFrame(rows)


# ============================================================
# 4. PREPARACIÓN DE VARIABLES PARA LOS ALGORITMOS
# ============================================================

def make_preprocessor(num_cols, cat_cols):

    # Lista que almacenará las transformaciones para cada tipo de variable.
    transformers = []

    # --------------------------------------------------------
    # Preparación de variables numéricas
    # --------------------------------------------------------

    if num_cols:

        # Si existen valores faltantes, se sustituyen por la mediana.
        # Después, StandardScaler estandariza las variables:
        # media = 0 y desviación estándar = 1.
        num_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )

        # Registramos las transformaciones para las columnas numéricas.
        transformers.append(("num", num_pipe, num_cols))

    # --------------------------------------------------------
    # Preparación de variables categóricas
    # --------------------------------------------------------

    if cat_cols:

        # Los valores faltantes se reemplazan por la categoría más frecuente.
        # OneHotEncoder transforma categorías en columnas binarias.
        cat_pipe = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

        # Registramos las transformaciones para las columnas categóricas.
        transformers.append(("cat", cat_pipe, cat_cols))

    # ColumnTransformer aplica cada transformación a sus columnas.
    # Las columnas que no se seleccionaron se descartan.
    return ColumnTransformer(transformers, remainder="drop")


# ============================================================
# 5. FUNCIÓN PARA CALCULAR EL COEFICIENTE DE SILUETA
# ============================================================

def safe_silhouette(X, labels, noise_label=None):

    # Convertimos las etiquetas de los grupos en un arreglo de NumPy.
    labels = np.asarray(labels)

    # Inicialmente consideramos todos los registros.
    mask = np.ones(len(labels), dtype=bool)

    # En DBSCAN, la etiqueta -1 representa puntos identificados como ruido.
    # Los excluimos para evaluar la calidad de los grupos encontrados.
    if noise_label is not None:
        mask = labels != noise_label

    # Seleccionamos los registros y etiquetas que se evaluarán.
    xx, yy = X[mask], labels[mask]

    # La silueta necesita al menos dos grupos y más registros que grupos.
    # Si no se cumplen estas condiciones, devolvemos NaN.
    if (
        len(set(yy)) < 2
        or len(yy) <= len(set(yy))
        or len(yy) < 3
    ):
        return np.nan

    try:

        # Calculamos el coeficiente de silueta.
        # El resultado está entre -1 y 1.
        # Se utiliza una muestra máxima de 1,500 registros para
        # reducir el tiempo de procesamiento en bases grandes.
        return float(
            silhouette_score(
                xx,
                yy,
                sample_size=min(1500, len(yy)),
                random_state=42
            )
        )

    # Si ocurre un error durante el cálculo, devolvemos NaN.
    except Exception:
        return np.nan


# ============================================================
# 6. PREPARACIÓN DE DATOS PARA K-PROTOTYPES
# ============================================================

def prep_kproto(frame, num_cols, cat_cols):

    # Seleccionamos únicamente las variables numéricas y categóricas.
    z = frame[num_cols + cat_cols].copy()

    # --------------------------------------------------------
    # Normalización de variables numéricas
    # --------------------------------------------------------

    for c in num_cols:

        # Aseguramos que la columna tenga valores numéricos.
        z[c] = pd.to_numeric(z[c], errors="coerce")

        # Reemplazamos los valores faltantes por la mediana.
        z[c] = z[c].fillna(z[c].median())

        # Calculamos la desviación estándar.
        sd = z[c].std()

        # Estandarizamos cada variable numérica.
        # Esto evita que una variable con valores grandes domine
        # el cálculo de distancias.
        z[c] = (
            (z[c] - z[c].mean()) /
            (sd if sd and np.isfinite(sd) else 1)
        )

    # --------------------------------------------------------
    # Preparación de variables categóricas
    # --------------------------------------------------------

    for c in cat_cols:

        # Convertimos los valores faltantes en "Desconocido".
        # Después convertimos todas las categorías a texto,
        # formato requerido para K-Prototypes.
        z[c] = (
            z[c]
            .astype("object")
            .where(z[c].notna(), "Desconocido")
            .astype(str)
        )

    # K-Prototypes recibe una matriz de objetos que contiene
    # tanto variables numéricas como categóricas.
    return z.to_numpy(dtype=object)


# ============================================================
# 7. EJECUCIÓN Y COMPARACIÓN DE LAS CINCO TÉCNICAS
# ============================================================

def run_comparison(
    df,
    num_cols,
    cat_cols,
    k,
    eps,
    min_samples,
    gamma,
    seed
):

    # Preparamos los datos para los algoritmos que trabajan
    # sobre variables numéricas transformadas.
    pre = make_preprocessor(num_cols, cat_cols)

    # Aplicamos las transformaciones seleccionadas.
    X = pre.fit_transform(df[num_cols + cat_cols])

    # Diccionario para guardar las etiquetas asignadas por cada algoritmo.
    labels = {}

    # Diccionario para guardar información de los modelos.
    models = {}

    # --------------------------------------------------------
    # ALGORITMO 1: K-MEANS
    # --------------------------------------------------------

    # Divide los datos en K grupos buscando minimizar la distancia
    # entre los clientes y el centro de su respectivo grupo.
    labels["K-Means"] = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=seed
    ).fit_predict(X)

    models["K-Means"] = "KMeans"

    # --------------------------------------------------------
    # ALGORITMO 2: CLUSTERING JERÁRQUICO
    # --------------------------------------------------------

    # Agrupa progresivamente los registros.
    # Ward busca minimizar el incremento de la variabilidad
    # dentro de los grupos al unir observaciones.
    labels["Jerárquico (Ward)"] = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    ).fit_predict(X)

    models["Jerárquico (Ward)"] = "AgglomerativeClustering"

    # --------------------------------------------------------
    # ALGORITMO 3: DBSCAN
    # --------------------------------------------------------

    # Identifica grupos a partir de la densidad de los registros.
    # eps define la distancia máxima para considerar vecinos.
    # min_samples define cuántos puntos se necesitan para formar
    # una región densa.
    # Los registros que no pertenecen a un grupo reciben etiqueta -1.
    labels["DBSCAN"] = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit_predict(X)

    models["DBSCAN"] = "DBSCAN"

    # --------------------------------------------------------
    # ALGORITMO 4: GAUSSIAN MIXTURE MODEL (GMM)
    # --------------------------------------------------------

    # GMM supone que los datos se generan a partir de una mezcla
    # de distribuciones gaussianas.
    # Permite representar grupos con diferentes formas y tamaños.
    # covariance_type="full" permite matrices de covarianza completas.
    labels["GMM"] = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        reg_covar=1e-6
    ).fit_predict(X)

    models["GMM"] = "GaussianMixture"

    # --------------------------------------------------------
    # ALGORITMO 5: K-PROTOTYPES
    # --------------------------------------------------------

    # K-Prototypes necesita al menos una variable categórica.
    if not cat_cols:
        raise ValueError(
            "Selecciona al menos una variable categórica para K-Prototypes."
        )

    # Preparamos los datos conservando sus tipos numéricos y categóricos.
    kp_data = prep_kproto(df, num_cols, cat_cols)

    # Identificamos las posiciones de las columnas categóricas.
    # Las numéricas aparecen primero y las categóricas después.
    cat_idx = list(
        range(len(num_cols), len(num_cols) + len(cat_cols))
    )

    # Inicializamos K-Prototypes.
    # gamma controla el peso relativo de las diferencias categóricas
    # frente a las numéricas.
    kp = KPrototypes(
        n_clusters=k,
        init="Cao",
        n_init=5,
        gamma=gamma,
        random_state=seed,
        verbose=0
    )

    # Entrenamos el modelo y obtenemos el segmento de cada cliente.
    labels["K-Prototypes"] = kp.fit_predict(
        kp_data,
        categorical=cat_idx
    )

    models["K-Prototypes"] = "KPrototypes"

    # --------------------------------------------------------
    # RESUMEN DE RESULTADOS
    # --------------------------------------------------------

    results = []

    # Recorremos los resultados de cada algoritmo.
    for name, lab in labels.items():

        # Identificamos las etiquetas de los grupos.
        unique = set(lab)

        # En DBSCAN, contamos cuántos clientes fueron clasificados como ruido.
        n_noise = (
            int(np.sum(np.asarray(lab) == -1))
            if name == "DBSCAN" else 0
        )

        # Contamos los grupos reales, sin incluir el ruido.
        n_groups = len(unique - {-1})

        # Calculamos cuántos clientes hay en cada grupo.
        sizes = pd.Series(lab).value_counts().sort_index()

        # Guardamos las métricas de cada técnica.
        results.append({

            "Técnica": name,

            "Grupos (sin ruido)": n_groups,

            "Clientes en ruido": n_noise,

            # Mostramos el tamaño de cada grupo.
            "Tamaños": ", ".join(
                f"{idx}: {count}"
                for idx, count in sizes.items()
            ),

            # Calculamos el coeficiente de silueta.
            "Silueta": safe_silhouette(
                X,
                lab,
                -1 if name == "DBSCAN" else None
            )
        })

    # Convertimos el resumen en una tabla.
    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # COMPARACIÓN DE COINCIDENCIA ENTRE ALGORITMOS
    # --------------------------------------------------------

    # Recuperamos los nombres de las técnicas utilizadas.
    names = list(labels)

    agreements = []

    # Comparamos cada algoritmo con los siguientes.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            agreements.append({

                "Técnica 1": names[i],

                "Técnica 2": names[j],

                # ARI mide qué tanto coinciden dos particiones.
                # 1 significa coincidencia perfecta.
                # 0 representa coincidencia similar a la aleatoria.
                # Puede tomar valores negativos si hay desacuerdo.
                "ARI (coincidencia)": adjusted_rand_score(
                    labels[names[i]],
                    labels[names[j]]
                )
            })

    # Devolvemos:
    # 1. Resumen de métricas.
    # 2. Comparación ARI.
    # 3. Etiquetas asignadas por cada algoritmo.
    # 4. Matriz transformada utilizada para los modelos.
    return (
        results_df,
        pd.DataFrame(agreements),
        labels,
        X
    )


# ============================================================
# 8. CREACIÓN DE LAS PESTAÑAS DE LA APLICACIÓN
# ============================================================

# La aplicación tendrá dos secciones:
# - Visor de grupos: analiza una segmentación existente.
# - Comparador: ejecuta y compara cinco algoritmos.
tab1, tab2 = st.tabs([
    "📊 Visor de grupos",
    "🧪 Comparador de técnicas"
])


# ============================================================
# 9. PRIMERA PESTAÑA: VISOR DE GRUPOS
# ============================================================

with tab1:

    st.subheader("Visor de grupos")

    # Permite cargar un archivo CSV que ya contenga una columna
    # con la clasificación o segmento de cada cliente.
    up = st.file_uploader(
        "Sube el CSV con los clientes y su columna de grupo",
        type="csv",
        key="viewer"
    )

    # El análisis se realiza únicamente si se cargó un archivo.
    if up is not None:

        # Leemos el CSV con Pandas.
        data = pd.read_csv(up)

        # Mostramos una vista previa de los primeros registros.
        st.write("Vista previa", data.head())

        # El usuario selecciona la columna que identifica el segmento.
        # Si existe una columna llamada Cluster, la seleccionamos por defecto.
        group_col = st.selectbox(
            "Columna que identifica el grupo",
            data.columns,
            index=(
                list(data.columns).index("Cluster")
                if "Cluster" in data.columns else 0
            )
        )

        # Buscamos posibles columnas relacionadas con gastos.
        spend_candidates = [
            c for c in data.columns
            if any(
                w in c.lower()
                for w in ["gasto", "spend", "mnt"]
            )
        ]

        # El usuario selecciona la variable de gasto.
        # Si encontramos una columna candidata, la seleccionamos por defecto.
        spend_col = st.selectbox(
            "Variable de gasto",
            ["(No mostrar)"] + list(data.columns),
            index=(
                1 + data.columns.get_loc(spend_candidates[0])
                if spend_candidates else 0
            )
        )

        # Si seleccionó No mostrar, no calculamos métricas de gasto.
        spend_col = (
            None if spend_col == "(No mostrar)" else spend_col
        )

        # Identificamos las columnas numéricas para utilizarlas
        # como ejes en el gráfico de dispersión.
        numeric = data.select_dtypes(
            include=np.number
        ).columns.tolist()

        # Selección de la variable para el eje X.
        xcol = (
            st.selectbox("Eje X", numeric, index=0)
            if numeric else None
        )

        # Selección de la variable para el eje Y.
        # Excluimos la variable elegida para X.
        ycol = (
            st.selectbox(
                "Eje Y",
                [c for c in numeric if c != xcol],
                index=0
            )
            if len(numeric) > 1 else None
        )

        # Generamos la tabla con el perfil de cada segmento.
        prof = profile_table(
            data,
            group_col,
            spend_col
        )

        # Dividimos el espacio en dos columnas para mostrar indicadores.
        c1, c2 = st.columns(2)

        # Indicador del total de clientes.
        c1.metric("Clientes", f"{len(data):,}")

        # Indicador de segmentos identificados.
        c2.metric(
            "Segmentos",
            data[group_col].nunique()
        )

        # Mostramos el perfil de los segmentos.
        st.dataframe(
            prof,
            use_container_width=True
        )

        # Si se seleccionaron dos variables numéricas,
        # construimos un gráfico de dispersión.
        if xcol and ycol:

            # Cada color representa un segmento diferente.
            fig = px.scatter(
                data,
                x=xcol,
                y=ycol,
                color=data[group_col].astype(str),

                # Si existe una columna identificadora,
                # la mostramos al pasar el cursor por un punto.
                hover_data=[
                    c for c in ["id", "ID", "customer_id"]
                    if c in data.columns
                ],

                title="Distribución de clientes por segmento"
            )

            # Mostramos el gráfico en Streamlit.
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Permite descargar el perfil de segmentos como CSV.
        st.download_button(
            "Descargar perfil de segmentos",
            prof.to_csv(index=False).encode("utf-8"),
            "perfil_segmentos.csv",
            "text/csv"
        )


# ============================================================
# 10. SEGUNDA PESTAÑA: COMPARADOR DE TÉCNICAS
# ============================================================

with tab2:

    st.subheader("Comparador de técnicas")

    # Permite cargar la base original de clientes.
    up2 = st.file_uploader(
        "Sube la base original de clientes (CSV)",
        type="csv",
        key="compare"
    )

    if up2 is not None:

        # Leemos los datos originales.
        data = pd.read_csv(up2)

        # Mostramos una vista previa.
        st.write("Vista previa", data.head())

        # Identificamos automáticamente las variables numéricas.
        numeric = data.select_dtypes(
            include=np.number
        ).columns.tolist()

        # Identificamos las variables categóricas.
        categorical = data.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        # Buscamos variables numéricas relacionadas con ingreso,
        # gasto, edad o cantidad de hijos para sugerirlas por defecto.
        default_num = [
            c for c in numeric
            if any(
                w in c.lower()
                for w in [
                    "ingreso", "gasto", "edad",
                    "hijo", "income", "spend", "age"
                ]
            )
        ]

        # Buscamos variables categóricas como educación y estado civil.
        default_cat = [
            c for c in categorical
            if any(
                w in c.lower()
                for w in [
                    "educ", "estado", "civil", "marital"
                ]
            )
        ]

        # El usuario selecciona las variables numéricas que utilizarán
        # los algoritmos de segmentación.
        num_cols = st.multiselect(
            "Variables numéricas (comunes a las técnicas)",
            numeric,
            default=default_num[:4]
        )

        # El usuario selecciona las variables categóricas.
        # Estas variables se utilizan especialmente en K-Prototypes.
        cat_cols = st.multiselect(
            "Variables categóricas (para K-Prototypes)",
            categorical,
            default=default_cat[:2]
        )

        # Identificamos posibles columnas relacionadas con gasto.
        spend_candidates = [
            c for c in numeric
            if any(
                w in c.lower()
                for w in ["gasto", "spend", "mnt"]
            )
        ]

        # Selección de la variable utilizada para calcular
        # el gasto promedio y el porcentaje del gasto.
        spend_col = st.selectbox(
            "Variable de gasto para perfiles",
            ["(No mostrar)"] + numeric,
            index=(
                1 + numeric.index(spend_candidates[0])
                if spend_candidates else 0
            )
        )

        # Parámetros generales de los algoritmos.
        # K define el número de grupos para K-Means, Ward, GMM
        # y K-Prototypes.
        k = st.slider(
            "Número de grupos (K)",
            2, 8, 3
        )

        # eps determina la distancia máxima entre vecinos en DBSCAN.
        eps = st.slider(
            "DBSCAN: eps",
            0.1, 5.0, 0.8, 0.1
        )

        # min_samples determina el número mínimo de puntos
        # necesarios para formar una región densa en DBSCAN.
        min_samples = st.slider(
            "DBSCAN: min_samples",
            2, 30, 8
        )

        # gamma controla el peso de las variables categóricas
        # frente a las numéricas en K-Prototypes.
        gamma = st.slider(
            "K-Prototypes: gamma",
            0.1, 5.0, 1.0, 0.1
        )

        # Semilla aleatoria para facilitar la reproducibilidad.
        seed = st.number_input(
            "Semilla aleatoria",
            0, 9999, 42
        )

        # El análisis se ejecuta cuando el usuario presiona el botón.
        if st.button(
            "Ejecutar comparación",
            type="primary"
        ):

            # Validamos que existan variables numéricas.
            if not num_cols:
                st.error(
                    "Selecciona al menos una variable numérica."
                )

            # Validamos que exista al menos una variable categórica.
            elif not cat_cols:
                st.error(
                    "Selecciona al menos una variable categórica para K-Prototypes."
                )

            else:

                try:

                    # Indicamos que los modelos se están entrenando.
                    with st.spinner(
                        "Entrenando las cinco técnicas..."
                    ):

                        # Ejecutamos los cinco algoritmos y recuperamos
                        # sus métricas, etiquetas y datos transformados.
                        summary, agreement, labels, X = run_comparison(
                            data,
                            num_cols,
                            cat_cols,
                            k,
                            eps,
                            min_samples,
                            gamma,
                            int(seed)
                        )

                    # ------------------------------------------------
                    # VISUALIZACIÓN DE MÉTRICAS
                    # ------------------------------------------------

                    st.markdown("#### Resultados")

                    # Mostramos cantidad de grupos, ruido,
                    # tamaño de grupos y coeficiente de silueta.
                    st.dataframe(
                        summary.round({"Silueta": 3}),
                        use_container_width=True
                    )

                    # Explicamos cómo interpretar las métricas.
                    st.caption(
                        "La silueta de DBSCAN se calcula excluyendo "
                        "los puntos etiquetados como ruido. ARI compara "
                        "particiones; 1 significa coincidencia perfecta "
                        "y cerca de 0 indica coincidencia similar al azar."
                    )

                    # ------------------------------------------------
                    # COMPARACIÓN ARI
                    # ------------------------------------------------

                    st.markdown(
                        "#### Coincidencia entre técnicas (ARI)"
                    )

                    # Mostramos qué tanto coinciden los segmentos
                    # generados por los diferentes algoritmos.
                    st.dataframe(
                        agreement.round(3),
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # PERFIL DE UNA TÉCNICA
                    # ------------------------------------------------

                    # El usuario selecciona el algoritmo que desea analizar.
                    selected = st.selectbox(
                        "Ver perfil de una técnica",
                        list(labels.keys())
                    )

                    # Creamos una copia de la base original.
                    out = data.copy()

                    # Agregamos la columna con el segmento asignado
                    # por el algoritmo seleccionado.
                    out["segmento"] = labels[selected]

                    # Determinamos si se mostrará la variable de gasto.
                    spend = (
                        None
                        if spend_col == "(No mostrar)"
                        else spend_col
                    )

                    # Calculamos el perfil comercial de los grupos.
                    st.dataframe(
                        profile_table(
                            out,
                            "segmento",
                            spend
                        ),
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # DESCARGA DE RESULTADOS
                    # ------------------------------------------------

                    # El usuario puede descargar la base original
                    # junto con la columna de segmentos asignados.
                    st.download_button(
                        "Descargar clientes con segmentos",

                        out.to_csv(
                            index=False
                        ).encode("utf-8"),

                        f"segmentos_{selected.lower().replace(' ', '_')}.csv",

                        "text/csv"
                    )

                # Si ocurre un error, mostramos su descripción
                # en la aplicación para facilitar su diagnóstico.
                except Exception as e:
                    st.error(
                        f"No se pudo ejecutar la comparación: {e}"
                    )

    else:

        # Mensaje inicial cuando todavía no se ha cargado un archivo.
        st.info(
            "Carga un CSV para seleccionar variables y comparar "
            "las cinco técnicas."
        )


# ============================================================
# 11. MENSAJE FINAL DE RECOMENDACIONES
# ============================================================

st.divider()

# Recomendaciones generales para utilizar correctamente la aplicación.
st.caption(
    "Consejo: excluye identificadores de los modelos, revisa valores "
    "faltantes y estandariza variables numéricas. La silueta y el ARI "
    "son indicadores técnicos; valida siempre la utilidad comercial "
    "con marketing."
)
