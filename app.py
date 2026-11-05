import os, json, joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='Billboard Top 1', layout='centered')
st.title('Predicción de éxito en Billboard - Cristopher Flores - ISC - Campus Comayagua')

@st.cache_resource
def cargar_modelo():
    # El archivo subido real se llama 'modelo' pero probaremos los nombres comunes.
    # Segun el log de joblib (pkl source), el dict tiene 'modelo' y 'columnas'.
    pkl = 'modelo'
    if not os.path.exists(pkl):
        if os.path.exists('decision_tree_billboard.pkl'):
            pkl = 'decision_tree_billboard.pkl'
        elif os.path.exists('modelo.pkl'):
            pkl = 'modelo.pkl'
    
    meta = 'model_metadata.json'
    
    # El archivo es un diccionario que contiene {'modelo': <objeto_arbol>, 'columnas': <lista>}
    data_dict = joblib.load(pkl)
    
    # Extraemos el modelo
    if isinstance(data_dict, dict) and 'modelo' in data_dict:
        modelo = data_dict['modelo']
    else:
        modelo = data_dict # Por si en otra version se guarda directo
        
    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base():
    csv = 'billboard (2).csv' if os.path.exists('billboard (2).csv') else 'billboard.csv'
    df = pd.read_csv(csv)
    # Limpieza básica para evitar errores con los selectores de Streamlit
    df = df.dropna(subset=['top', 'artist_type', 'tempo', 'genre', 'mood'])
    for col in ['durationSeg', 'anioNacimiento']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['top'] = df['top'].astype(int)
    return df

modelo, metadata = cargar_modelo()
df = cargar_base()

mapa = {int(k): v for k, v in metadata['mapa_top'].items()}

st.caption(metadata['nombre_modelo'])

with st.form('datos'):
    col1, col2 = st.columns(2)

    durationSeg = col1.number_input('Duración (Segundos)', min_value=0.0, max_value=1000.0, value=200.0)
    anioNacimiento = col2.number_input('Año de Nacimiento', min_value=1900.0, max_value=2026.0, value=1990.0)

    artist_type = col1.selectbox('Tipo de Artista', sorted(df['artist_type'].astype(str).unique()))
    tempo = col2.selectbox('Tempo', sorted(df['tempo'].astype(str).unique()))

    genre = col1.selectbox('Género', sorted(df['genre'].astype(str).unique()))
    mood = col2.selectbox('Estado de Ánimo', sorted(df['mood'].astype(str).unique()))

    enviar = st.form_submit_button('Evaluar')

if enviar:
    cliente = pd.DataFrame([{
        'durationSeg': durationSeg,
        'anioNacimiento': anioNacimiento,
        'artist_type': artist_type,
        'tempo': tempo,
        'genre': genre,
        'mood': mood
    }])

    # Transformar a las variables dummy que requiere el modelo
    cliente_dummy = pd.get_dummies(cliente)
    
    # Cargar las columnas que espera el modelo (guardadas en sus features o metadata)
    # Segun el volcado de joblib, las variables estan en modelo.feature_names_in_
    columnas_modelo = list(modelo.feature_names_in_)
    
    # Asegurar que el dataframe final tenga exactamente las mismas columnas que el modelo
    for col in columnas_modelo:
        if col not in cliente_dummy.columns:
            cliente_dummy[col] = 0
            
    # Ordenar las columnas para que coincidan exactamente
    cliente_final = cliente_dummy[columnas_modelo]

    # Predecir
    prediccion = int(modelo.predict(cliente_final)[0])
    resultado_str = mapa.get(prediccion, 'No definido')

    st.subheader(f'Predicción: {resultado_str}')

    api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))

    if api_key:
        prompt = f'''
        Actúa como un experto en la industria musical y análisis de datos.

        Explica brevemente el resultado de la predicción y brinda
        3 recomendaciones prudentes para que una canción mejore sus probabilidades de alcanzar el Top 1.

        Datos de la canción:
        Duración={durationSeg} seg
        Año de nacimiento={anioNacimiento}
        Tipo de artista={artist_type}
        Tempo={tempo}
        Género={genre}
        Estado de ánimo={mood}

        Resultado de la predicción:
        Llegó al Top 1={resultado_str}
        '''

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            respuesta = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista de datos y experto en la industria musical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            texto = respuesta.choices[0].message.content
            st.info(texto)

        except Exception as e:
            st.warning(f'Error con Groq: {e}')
    else:
        st.warning('Agregue GROQ_API_KEY en los secretos de Streamlit.')

st.divider()
st.write('Vista rápida de la base principal')
st.dataframe(df.head(20), use_container_width=True)
