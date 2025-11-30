import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# -----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="EcoStats: Distribuciones Muestrales",
    layout="wide",
    page_icon="📊"
)

# -----------------------------
# ESTILOS CSS PERSONALIZADOS (Estilo EcoStats)
# -----------------------------
st.markdown("""
<style>
    /* Tarjetas de métricas */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2E8B57; /* Verde EcoStats */
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    
    /* Títulos */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1C1C1C;
    }
    
    .highlight {
        color: #2E8B57;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCIONES DE GENERACIÓN DE DATOS
# -----------------------------


def generar_poblacion_multimodal(n_puntos=10000, media1=10, std1=2, media2=20, std2=2, mix=0.5):
    """Genera una distribución bimodal mezclando dos normales."""
    n1 = int(n_puntos * mix)
    n2 = n_puntos - n1
    data1 = np.random.normal(media1, std1, n1)
    data2 = np.random.normal(media2, std2, n2)
    return np.concatenate([data1, data2])


def obtener_medias_muestrales(poblacion, n_muestra, n_simulaciones):
    """Extrae muestras de tamaño n y calcula sus medias."""
    medias = []
    for _ in range(n_simulaciones):
        muestra = np.random.choice(poblacion, size=n_muestra, replace=True)
        medias.append(np.mean(muestra))
    return np.array(medias)


# -----------------------------
# BARRA LATERAL (Menú)
# -----------------------------
with st.sidebar:
    st.markdown("## 📊 Distribuciones")
    st.markdown("Simulador Interactivo")

    selected = option_menu(
        menu_title="Menú Principal",
        options=["Inicio", "Simulación", "Teoría", "Acerca de"],
        icons=["house", "activity", "book", "info-circle"],
        menu_icon="cast",
        default_index=1,  # Por defecto abre en simulación
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuración Global")
    st.info("Usa los controles dentro de la pestaña 'Simulación' para modificar el experimento.")

# -----------------------------
# PÁGINA: INICIO
# -----------------------------
if selected == "Inicio":
    st.title("📊 Explorador de Distribuciones Multimodales")
    st.markdown("""
    Bienvenido a este dashboard interactivo diseñado para entender el comportamiento de las **Distribuciones Muestrales**.
    
    ### ¿Qué vas a descubrir?
    1.  **Poblaciones Multimodales**: ¿Qué pasa cuando tus datos originales tienen dos "picos" o modas?
    2.  **El poder de 'n'**: Cómo el tamaño de la muestra transforma la forma de los datos.
    3.  **Teorema del Límite Central (TLC)**: Verás en tiempo real cómo surge la "Campana de Gauss" incluso desde datos caóticos.
    
    Ve a la sección **Simulación** para empezar a jugar con los datos.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Illustration_Central_Limit_Theorem.svg/1200px-Illustration_Central_Limit_Theorem.svg.png", caption="Concepto del TLC")

# -----------------------------
# PÁGINA: SIMULACIÓN (CORE)
# -----------------------------
elif selected == "Simulación":
    st.title("🧪 Laboratorio de Muestreo")

    # --- CONTROLES SUPERIORES ---
    st.markdown("### 1. Configura tu Población (La realidad)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mu1 = st.number_input("Media Pico 1", value=10.0, step=1.0)
    with c2:
        sigma1 = st.number_input(
            "Desv. Std Pico 1", value=2.0, min_value=0.1, step=0.1)
    with c3:
        mu2 = st.number_input("Media Pico 2", value=22.0, step=1.0)
    with c4:
        sigma2 = st.number_input(
            "Desv. Std Pico 2", value=3.0, min_value=0.1, step=0.1)

    st.markdown("---")

    # --- CONTROLES DE MUESTRA (LO QUE PEDISTE) ---
    st.markdown("### 2. Configura el Muestreo (El experimento)")

    col_n, col_sim = st.columns([2, 1])

    with col_n:
        n_size = st.slider(
            "📏 Tamaño de la muestra (n)",
            min_value=1,
            max_value=200,
            value=1,
            step=1,
            help="Este es el número de datos que tomamos en cada 'paquete' para sacar un promedio."
        )
        if n_size == 1:
            st.warning(
                "⚠️ Con n=1, la distribución muestral es idéntica a la población original.")
        elif n_size >= 30:
            st.success(
                "✅ Con n >= 30, el efecto del Teorema del Límite Central suele ser muy visible.")

    with col_sim:
        n_sims = st.selectbox(
            "🔄 Número de simulaciones",
            options=[100, 500, 1000, 5000, 10000],
            index=2,
            help="Cuántas veces repetimos el proceso de tomar 'n' datos y promediarlos."
        )

    # --- LÓGICA DE CÁLCULO ---
    # 1. Generar Población
    poblacion = generar_poblacion_multimodal(
        media1=mu1, std1=sigma1, media2=mu2, std2=sigma2)

    # 2. Generar Distribución Muestral
    muestras = obtener_medias_muestrales(poblacion, n_size, n_sims)

    # --- VISUALIZACIÓN ---
    st.markdown("---")

    # GRÁFICO 1: POBLACIÓN ORIGINAL
    st.subheader("1️⃣ Población Original (Distribución Multimodal)")
    st.caption("Así se ven todos los datos individuales mezclados.")

    fig_pop = px.histogram(
        x=poblacion,
        nbins=100,
        opacity=0.7,
        color_discrete_sequence=['#636EFA'],
        labels={'x': 'Valor', 'count': 'Frecuencia'}
    )
    fig_pop.update_layout(
        title_text=f"Histograma de la Población (N=10,000)",
        bargap=0.1,
        template="plotly_white",
        height=300
    )
    # Añadir densidad suave (KDE) simulada visualmente
    st.plotly_chart(fig_pop, use_container_width=True)

    # GRÁFICO 2: DISTRIBUCIÓN DE MEDIAS
    st.subheader(f"2️⃣ Distribución de las Medias Muestrales (n = {n_size})")
    st.caption(
        f"Aquí mostramos el histograma de {n_sims} promedios calculados.")

    fig_sample = go.Figure()

    # Histograma
    fig_sample.add_trace(go.Histogram(
        x=muestras,
        name='Medias Muestrales',
        opacity=0.75,
        marker_color='#2E8B57',  # Verde estilo EcoStats
        histnorm='probability density'
    ))

    # Curva Normal Teórica (Superpuesta)
    mu_teorica = np.mean(poblacion)
    sigma_teorica = np.std(poblacion) / np.sqrt(n_size)
    x_range = np.linspace(min(muestras), max(muestras), 1000)
    pdf = stats.norm.pdf(x_range, mu_teorica, sigma_teorica)

    fig_sample.add_trace(go.Scatter(
        x=x_range,
        y=pdf,
        mode='lines',
        name=f'Normal Teórica (CLT)',
        line=dict(color='red', width=3, dash='dash')
    ))

    fig_sample.update_layout(
        title=f"Distribución de Medias (n={n_size}) vs Curva Normal",
        xaxis_title="Valor Promedio",
        yaxis_title="Densidad",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig_sample, use_container_width=True)

    # --- MÉTRICAS COMPARATIVAS ---
    st.markdown("### 📊 Estadísticas Comparativas")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Media Población (μ)</div>
            <div class="metric-value">{np.mean(poblacion):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        # CORRECCIÓN AQUÍ: Usamos {{x}} para que Python no busque una variable 'x'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Media de Medias ($\overline{{x}}$)</div>
            <div class="metric-value">{np.mean(muestras):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Desv. Std Población (σ)</div>
            <div class="metric-value" style="color: #636EFA;">{np.std(poblacion):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Error Estándar ($SE$)</div>
            <div class="metric-value" style="color: #636EFA;">{np.std(muestras):.2f}</div>
            <small style="color:gray">Teórico: {np.std(poblacion)/np.sqrt(n_size):.2f}</small>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# PÁGINA: TEORÍA
# -----------------------------
elif selected == "Teoría":
    st.title("📚 Fundamentos Teóricos")

    st.markdown("""
    ### Distribución Multimodal
    Una distribución multimodal es aquella que tiene dos o más "modas" o picos. En nuestro ejemplo, simulamos esto combinando dos distribuciones normales con diferentes medias. Esto es común en la naturaleza (ej. alturas de hombres y mujeres combinadas).
    
    ### El Teorema del Límite Central (TLC)
    Este teorema es la razón por la que el segundo gráfico se vuelve una campana perfecta cuando aumentas **n**.
    
    Establece que:
    > Si tomas muestras de tamaño $n$ suficientemente grande de **cualquier** población (sin importar si es bimodal, plana o extraña), la distribución de las medias de esas muestras se aproximará a una **Distribución Normal**.
    
    #### Fórmulas Clave:
    Si la población tiene media $\mu$ y desviación estándar $\sigma$:
    
    1.  **Media de medias:** $\mu_{\bar{x}} \approx \mu$
    2.  **Error Estándar (nueva desviación):** $\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$
    
    Observa en la pestaña "Simulación" cómo el valor de **Error Estándar** disminuye drásticamente a medida que mueves el slider de $n$ hacia la derecha.
    """)

elif selected == "Acerca de":
    st.markdown("## 👨‍💻 Sobre este Dashboard")
    st.write("Creado para experimentar con conceptos de estadística inferencial usando Python y Streamlit.")
    st.info("Desarrollado con el estilo de **EcoStats**.")
