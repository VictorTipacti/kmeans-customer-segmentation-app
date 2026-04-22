import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Segmentador Pro",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# ESTILO PERSONALIZADO
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stDownloadButton>button {
    background-color: #008CBA;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🚀 Segmentación Inteligente de Datos")
st.markdown("Analiza, segmenta y visualiza tus datos de forma interactiva.")

# -----------------------------
# SESSION
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "centroids" not in st.session_state:
    st.session_state.centroids = None

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuración")

k = st.sidebar.slider("Número de Clusters", 2, 10, 3)

if st.sidebar.button("🔄 Reset"):
    st.session_state.data = None
    st.session_state.centroids = None

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Necesitas al menos 2 columnas numéricas")
    else:
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Variable X", numeric_cols)

        with col2:
            y_col = st.selectbox("Variable Y", numeric_cols)

        # -----------------------------
        # EJECUTAR
        # -----------------------------
        if st.button("🚀 Ejecutar Análisis"):

            data = df[[x_col, y_col]].dropna().copy()
            data = data.reset_index(drop=True)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(scaled)

            data["Cluster"] = labels

            st.session_state.data = data
            st.session_state.centroids = scaler.inverse_transform(model.cluster_centers_)

        # -----------------------------
        # RESULTADOS
        # -----------------------------
        if st.session_state.data is not None:

            data = st.session_state.data
            centroids = st.session_state.centroids

            st.subheader("📊 Visualización")

            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=data["Cluster"].astype(str),
                template="plotly_dark",
                size_max=10
            )

            fig.add_scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(size=18, symbol="x"),
                name="Centroides"
            )

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # MÉTRICAS VISUALES
            # -----------------------------
            st.subheader("📈 Resumen")

            colA, colB, colC = st.columns(3)

            colA.metric("Total datos", len(data))
            colB.metric("Clusters", k)
            colC.metric("Promedio X", round(data[x_col].mean(), 2))

            # -----------------------------
            # ANALISIS POR CLUSTER
            # -----------------------------
            st.subheader("🔍 Análisis por Cluster")

            cluster_sel = st.selectbox(
                "Selecciona un cluster",
                sorted(data["Cluster"].unique())
            )

            subset = data[data["Cluster"] == cluster_sel]

            st.dataframe(subset, use_container_width=True)

            # -----------------------------
            # DESCARGA
            # -----------------------------
            st.subheader("💾 Exportar")

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Descargar resultados",
                csv,
                "clusters.csv",
                "text/csv"
            )

            # -----------------------------
            # EXPLICACIÓN
            # -----------------------------
            st.subheader("📘 ¿Qué hace el modelo?")

            st.info(f"""
K-Means con K={k} agrupa datos similares.

✔ Minimiza distancia a centroides  
✔ Itera hasta estabilizarse  
✔ Genera segmentos claros
""")