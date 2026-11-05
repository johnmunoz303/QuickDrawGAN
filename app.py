import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
from gan import CATEGORY, Generator, Discriminator, load_data, train

st.set_page_config(page_title='QuickDraw · Laboratorio GAN', page_icon='✏️', layout='wide')
st.title('✏️ De ruido a dibujos · Laboratorio GAN')
st.caption('Categoría: smiley face / cara sonriente · Quick, Draw! · imágenes de 28 × 28 píxeles')
st.info('Esta GAN crea dibujos. El discriminador distingue real de generado; no adivina categorías como el juego original. Los resultados aparecen al entrenar, sin métricas simuladas.')
with st.sidebar:
    st.header('Configura el experimento')
    latent = st.select_slider('Dimensión del ruido z', [16, 32, 64, 128], value=64)
    epochs = st.slider('Épocas (recorridos del conjunto)', 1, 100, 20)
    limit = st.select_slider('Cantidad de dibujos reales', [1000, 3000, 5000, 10000], value=3000)
    batch = st.selectbox('Tamaño de lote', [32, 64, 128], index=1)
    lr_g = st.number_input('Tasa de aprendizaje G', min_value=0.00001, max_value=0.01, value=0.0002, format='%.5f')
    lr_d = st.number_input('Tasa de aprendizaje D', min_value=0.00001, max_value=0.01, value=0.0002, format='%.5f')
    seed = st.number_input('Semilla', 0, 99999, 42)
    start = st.button('Descargar datos y entrenar', type='primary')

learn, lab, weights, code = st.tabs(['1 · Entender las redes', '2 · Entrenar y generar', '3 · Inspeccionar valores', '4 · Código comentado'])
with learn:
    a, b = st.columns(2)
    with a:
        st.subheader('Generador G')
        st.write('Ruido → Dense(128) → Dense(256) → Dense(784) → dibujo 28×28. LeakyReLU entre capas y Tanh al final.')
        st.write('Aprende pesos y sesgos para producir dibujos que el discriminador considere reales.')
        st.metric('Parámetros aprendibles con esta dimensión z', sum(p.numel() for p in Generator(latent).parameters()))
    with b:
        st.subheader('Discriminador D')
        st.write('Dibujo (784) → Dense(256) → Dense(128) → Dense(1). LeakyReLU entre capas. La salida es un logit.')
        st.write('Sigmoid convierte el logit en una puntuación entre 0 y 1. No es una probabilidad calibrada de calidad.')
        st.metric('Parámetros aprendibles', sum(p.numel() for p in Discriminator().parameters()))
    st.subheader('Funciones de pérdida')
    st.latex(r'L_D=-\mathbb{E}_{x}[\log D(x)]-\mathbb{E}_{z}[\log(1-D(G(z)))]')
    st.latex(r'L_G=-\mathbb{E}_{z}[\log D(G(z))]')
    st.write('Aquí D representa sigmoid del logit. D minimiza la suma de dos medias BCE; G minimiza la BCE con etiqueta 1. Se alterna una actualización de D y una de G por lote.')
    st.write('Si D devuelve 0,5 para todo, L_D ≈ 1,386 y L_G ≈ 0,693. Esto no prueba que los dibujos sean buenos: también puede ocurrir si D no aprende. Observa variedad, trazos y pérdidas en conjunto.')
    st.subheader('Qué significa cada control')
    st.write('Dimensión z: cantidad de números aleatorios de entrada. Épocas: repeticiones del entrenamiento. Lote: ejemplos por actualización. Tasa de aprendizaje: tamaño del ajuste de pesos. Semilla: fija el azar. Adam usa β₁=0,5 y β₂=0,999. Los píxeles se normalizan de [0,255] a [-1,1].')
with lab:
    status = st.empty()
    metrics = st.empty()
    chart = st.empty()
    gallery = st.empty()
    st.caption('Galería con el mismo ruido durante el entrenamiento para comparar la evolución. Las imágenes pueden empezar como ruido y el modelo puede colapsar a dibujos similares.')
    if start:
        try:
            with st.spinner('Descargando dibujos reales de cara sonriente…'):
                data = load_data(limit)
            st.write('Ejemplos reales de entrenamiento')
            st.image([255 - v.reshape(28, 28) for v in data[:8]], width=84)
            def report(history, samples):
                status.write(f"Entrenando: época {len(history)} / {epochs} · categoría {CATEGORY}")
                metrics.dataframe(pd.DataFrame(history[-1:], columns=['época', 'G', 'D', 'D(real)', 'D(G(z))']), hide_index=True)
                chart.line_chart(pd.DataFrame(history).set_index('época')[['G', 'D']])
                gallery.image([1 - s for s in samples], width=84, clamp=True)
            g, d, history = train(data, epochs, batch, latent, lr_g, lr_d, seed, report)
            st.session_state.result = (g, d, history, latent, dict(category=CATEGORY, epochs=epochs, samples=len(data), batch=batch, lr_g=lr_g, lr_d=lr_d, seed=seed))
            status.success('Entrenamiento terminado. Puedes inspeccionar pesos y generar nuevos dibujos.')
        except Exception as exc:
            st.error(f'No se pudo completar el entrenamiento: {exc}. Comprueba la conexión y las dependencias, y vuelve a intentarlo.')
    if 'result' in st.session_state:
        g, d, history, trained_latent, config = st.session_state.result
        st.write('Configuración del último modelo entrenado', config)
        if not start:
            st.line_chart(pd.DataFrame(history).set_index('época')[['G', 'D']])
        if st.button('Generar 16 dibujos nuevos'):
            with torch.no_grad():
                fake = g(torch.randn(16, trained_latent))
                scores = d(fake).sigmoid().flatten().tolist()
            st.image([1 - s for s in ((fake.reshape(16,28,28)+1)/2).numpy()], caption=[f'Cara sonriente · D={s:.3f}' for s in scores], width=100, clamp=True)
        st.download_button('Descargar pérdidas CSV', pd.DataFrame(history).to_csv(index=False).encode('utf-8'), 'perdidas.csv', 'text/csv')
        buffer = io.BytesIO()
        torch.save(dict(generator=g.state_dict(), discriminator=d.state_dict(), latent=trained_latent, config=config, history=history), buffer)
        st.download_button('Descargar pesos y configuración', buffer.getvalue(), 'gan_smiley_face.pt')
with weights:
    if 'result' not in st.session_state:
        st.write('Entrena primero para mostrar valores aprendidos reales.')
    else:
        g, d, _, _, _ = st.session_state.result
        name = st.radio('Red a inspeccionar', ['Generador', 'Discriminador'], horizontal=True)
        model = g if name == 'Generador' else d
        rows = []
        for key, tensor in model.state_dict().items():
            v = tensor.detach().numpy()
            rows.append(dict(tensor=key, forma=str(v.shape), cantidad=v.size, media=float(v.mean()), desviación=float(v.std()), mínimo=float(v.min()), máximo=float(v.max())))
        st.dataframe(rows, hide_index=True)
        key = st.selectbox('Pesos o sesgos', list(model.state_dict()))
        st.write('Primeros 100 valores del tensor seleccionado')
        st.dataframe(model.state_dict()[key].detach().numpy().flatten()[:100])
with code:
    st.write('Las marcas AQUÍ SE CREA identifican ambas redes. PASO D y PASO G identifican cada actualización.')
    from pathlib import Path
    st.code(Path(__file__).with_name('gan.py').read_text(encoding='utf-8-sig'), language='python')
st.divider()
st.markdown('Fuentes: [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) · [Conceptos GAN de Google](https://developers.google.com/machine-learning/gan?hl=es-419). Datos: CC BY 4.0, Google Creative Lab y participantes de Quick, Draw!.')
