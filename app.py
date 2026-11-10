"""Aplicación educativa de clasificación visual con consulta taxonómica externa."""
from pathlib import Path
from io import BytesIO
import os
import re
from urllib.parse import urlparse
import joblib
import requests
import streamlit as st
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
NOMBRES = {
    'Cynodon dactylon': 'Grama de Bermuda',
    'Gliricidia sepium': 'Piñón cubano',
    'Megathyrsus maximus': 'Hierba de Guinea',
    'Paspalum notatum': 'Pasto bahía',
}
# Panicum maximum es un sinónimo taxonómico usado para Megathyrsus maximus.
ALIAS_ESPECIES = {
    'panicum maximum': 'Megathyrsus maximus',
    **{nombre.casefold(): nombre for nombre in NOMBRES},
}
URL_API_PLANTNET = 'https://my-api.plantnet.org/v2/identify/all'

st.set_page_config(page_title='Plantas forrajeras del Cibao', page_icon='🌿', layout='centered')
st.markdown('''
<style>
:root {
  --fondo: #f4f1e8;
  --papel: #fffefa;
  --verde: #194d45;
  --verde-claro: #e9f0eb;
  --terracota: #c45f48;
  --texto: #293431;
  --suave: #68746f;
  --borde: #d8ded7;
}
[data-testid="stAppViewContainer"] { background: var(--fondo); color: var(--texto); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] > div { padding-top: 2.4rem; }
.block-container { max-width: 980px; padding: 2.2rem 2rem 3.5rem; }
h1, h2, h3 { color: var(--verde) !important; letter-spacing: -0.025em; }
h1 { font-size: clamp(2.15rem, 4vw, 3.1rem) !important; line-height: 1.08 !important; margin-bottom: .8rem !important; }
p { color: var(--texto); line-height: 1.65; }
[data-testid="stCaptionContainer"] { color: var(--suave); line-height: 1.55; }
[data-testid="stAlert"] { border: 1px solid #d5e2d8; border-left: 5px solid var(--verde); border-radius: 14px; background: var(--verde-claro); color: var(--verde); box-shadow: 0 5px 18px rgba(32, 60, 49, .05); }
[data-testid="stFileUploader"] section { background: var(--papel); border: 1.5px dashed #9eafa2; border-radius: 16px; padding: 1.1rem; }
[data-testid="stFileUploader"] button { background: var(--verde); color: white; border: 0; border-radius: 9px; }
[data-testid="stButton"] button[kind="primary"] { background: var(--terracota); border: 1px solid var(--terracota); color: white; border-radius: 10px; font-weight: 650; padding: .62rem 1.15rem; box-shadow: 0 5px 12px rgba(196, 95, 72, .18); }
[data-testid="stButton"] button[kind="primary"]:hover { background: #a94d39; border-color: #a94d39; }
[data-testid="stDownloadButton"] button, [data-testid="stLinkButton"] a { background: var(--papel); color: var(--verde); border: 1px solid var(--borde); border-radius: 10px; font-weight: 600; }
[data-testid="stDownloadButton"] button:hover, [data-testid="stLinkButton"] a:hover { color: var(--terracota); border-color: var(--terracota); }
[data-testid="stExpander"] { background: var(--papel); border: 1px solid var(--borde); border-radius: 14px; box-shadow: 0 5px 18px rgba(32, 60, 49, .04); }
[data-testid="stTable"] { background: var(--papel); border-radius: 12px; overflow: hidden; }
hr { border-color: var(--borde); }
@media (max-width: 640px) {
  .block-container { padding: 1.4rem 1rem 2.5rem; }
  h1 { font-size: 2.05rem !important; }
}
</style>
''', unsafe_allow_html=True)
st.title('Plantas forrajeras del Cibao')
st.write('Identificación asistida por imágenes para las clases de Pastos y Forrajes del Politécnico José Antonio Castillo.')
st.info('Prototipo experimental. La validación interna del clasificador local obtuvo 46,9 % de exactitud media. Toda identificación automática es una candidata y requiere revisión docente.')
st.write('Clases locales: hierba de Guinea, grama de Bermuda, piñón cubano y pasto bahía.')
st.caption('Al pulsar «Analizar fotografía», la imagen se enviará a Pl@ntNet para buscar una candidata taxonómica y compararla con las cuatro clases. La imagen se convierte a JPEG y se eliminan sus metadatos EXIF antes del envío. No uses una imagen que no tengas autorización para compartir.')


def obtener_clave_plantnet():
    """Lee la clave de Streamlit Secrets o, en ejecución local, del entorno."""
    try:
        clave = st.secrets.get('PLANTNET_API_KEY', '')
    except Exception:
        clave = ''
    return clave or os.getenv('PLANTNET_API_KEY', '')


def normalizar_nombre_cientifico(nombre):
    """Devuelve el binomio para comparar nombres sin autoría taxonómica."""
    if not nombre:
        return ''
    partes = re.sub(r'[^A-Za-z\s-]', ' ', str(nombre)).split()
    return ' '.join(partes[:2]).casefold()


def identificar_con_plantnet(imagen, clave):
    """Consulta Pl@ntNet por POST y no expone la clave en errores de interfaz."""
    buffer = BytesIO()
    imagen.save(buffer, format='JPEG', quality=90, optimize=True)
    datos_imagen = buffer.getvalue()
    respuesta = requests.post(
        URL_API_PLANTNET,
        params={
            'api-key': clave,
            'lang': 'es',
            'nb-results': 5,
            'include-related-images': 'true',
        },
        files={'images': ('planta.jpg', datos_imagen, 'image/jpeg')},
        timeout=(10, 45),
    )
    if respuesta.status_code == 404:
        return None
    if respuesta.status_code == 429:
        raise RuntimeError('Pl@ntNet alcanzó el límite temporal de consultas. Inténtalo más tarde.')
    if respuesta.status_code == 401:
        raise RuntimeError('Pl@ntNet respondió HTTP 401: no autorizó la clave. Verifica que sea la clave privada de la API, copiada completa y sin el prefijo «api-key:». No compartas la clave.')
    if respuesta.status_code == 403:
        raise RuntimeError('Pl@ntNet respondió HTTP 403: la solicitud está bloqueada por la configuración de acceso de la clave. En los ajustes de Pl@ntNet revisa «Expose my API key» y las «Authorized IPs»; esta aplicación consulta desde el servidor, no desde el navegador.')
    if respuesta.status_code >= 500:
        raise RuntimeError('Pl@ntNet no está disponible en este momento. Inténtalo más tarde.')
    if not respuesta.ok:
        raise RuntimeError(f'Pl@ntNet respondió con el código {respuesta.status_code}.')
    try:
        resultado = respuesta.json()
    except ValueError as error:
        raise RuntimeError('Pl@ntNet devolvió una respuesta que no se pudo leer.') from error
    candidatos = resultado.get('results') or []
    return (resultado, candidatos) if candidatos else None


def especie_de_candidato(candidato, resultado):
    especie = (candidato.get('species') or {}).get('scientificNameWithoutAuthor')
    return especie or resultado.get('bestMatch', '')


def clase_del_candidato(nombre_cientifico):
    nombre = normalizar_nombre_cientifico(nombre_cientifico)
    return ALIAS_ESPECIES.get(nombre)


def mostrar_referencia_gbif(candidato):
    identificador = (candidato.get('gbif') or {}).get('id')
    if identificador:
        st.link_button('Consultar ficha taxonómica en GBIF', f'https://www.gbif.org/species/{identificador}')


def mostrar_imagenes_de_referencia(candidato):
    imagenes = candidato.get('images') or []
    imagenes_legibles = []
    for referencia in imagenes:
        direcciones = referencia.get('url') or {}
        if isinstance(direcciones, dict):
            url = direcciones.get('m') or direcciones.get('o') or direcciones.get('s')
        elif isinstance(direcciones, str):
            url = direcciones
        else:
            url = None
        if not isinstance(url, str) or not url:
            continue
        credito = referencia.get('citation') or referencia.get('author') or 'Autor no indicado en la respuesta'
        licencia = referencia.get('license') or 'Licencia no indicada en la respuesta'
        imagenes_legibles.append((url, f'{credito} · Pl@ntNet · {licencia}'))
    if imagenes_legibles:
        st.markdown('**Imágenes similares de referencia**')
        for url, credito in imagenes_legibles[:2]:
            try:
                direccion = urlparse(url)
                if direccion.scheme != 'https' or direccion.hostname != 'bs.plantnet.org':
                    raise ValueError('Dirección de imagen no reconocida.')
                respuesta = requests.get(url, timeout=(5, 15))
                respuesta.raise_for_status()
                if not respuesta.headers.get('Content-Type', '').lower().startswith('image/'):
                    raise ValueError('El enlace no devolvió una imagen.')
                with Image.open(BytesIO(respuesta.content)) as original:
                    original.load()
                    imagen = original.convert('RGB')
                st.image(imagen, caption=credito, width=380)
            except Exception:
                st.caption('No se pudo cargar esta imagen de referencia. La candidata taxonómica se muestra arriba para revisión.')


@st.cache_resource
def cargar_modelo_local():
    torch.set_num_threads(2)
    extractor = mobilenet_v3_small(weights=None)
    extractor.classifier = torch.nn.Identity()
    extractor.load_state_dict(torch.load(ROOT / 'modelos/extractor.pt', map_location='cpu', weights_only=True))
    extractor.eval()
    return extractor, joblib.load(ROOT / 'modelos/clasificador.joblib'), MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()


foto = st.file_uploader('Carga una fotografía de una sola planta', type=['jpg', 'jpeg', 'png'])
if foto is not None:
    try:
        if foto.size > 10 * 1024 * 1024:
            raise ValueError('La imagen debe pesar como máximo 10 MB.')
        with Image.open(BytesIO(foto.getvalue())) as origen:
            imagen = ImageOps.exif_transpose(origen).convert('RGB')
        st.image(imagen, caption='Fotografía cargada', width=380)
        if min(imagen.size) < 800:
            st.warning('La imagen tiene menos de 800 píxeles en uno de sus lados; Pl@ntNet indica que su rendimiento no está garantizado con imágenes de ese tamaño.')

        if st.button('Analizar fotografía', type='primary'):
            clave = obtener_clave_plantnet()
            if not clave:
                st.error('Falta configurar la clave API de Pl@ntNet. No se mostrará la predicción forzada del clasificador local porque solo conoce las cuatro clases del proyecto.')
                st.code('PLANTNET_API_KEY = "tu_clave_privada"', language='toml')
                st.caption('En Streamlit Cloud, añade PLANTNET_API_KEY en App settings → Secrets. En ejecución local, configúrala como variable de entorno. No la publiques en GitHub.')
            elif not (ROOT / 'modelos/clasificador.joblib').exists() or not (ROOT / 'modelos/extractor.pt').exists():
                st.error('Faltan los archivos entrenados del clasificador local. La consulta taxonómica no se ejecutó.')
            else:
                try:
                    with st.spinner('Consultando Pl@ntNet y comparando con las cuatro clases…'):
                        consulta = identificar_con_plantnet(imagen, clave)
                        extractor, clasificador, preparar = cargar_modelo_local()
                        with torch.inference_mode():
                            vector = extractor(preparar(imagen).unsqueeze(0)).numpy()
                        especie_local = clasificador.predict(vector)[0]

                    if consulta is None:
                        st.warning('Pl@ntNet no encontró una especie candidata para esta imagen. No se puede confirmar que pertenezca a alguna de las cuatro clases.')
                        st.caption('Prueba con una fotografía nítida donde se vea una parte diagnóstica de una sola planta. La predicción local no se presenta como identificación.')
                    else:
                        respuesta, candidatos = consulta
                        principal = candidatos[0]
                        nombre = especie_de_candidato(principal, respuesta)
                        clase = clase_del_candidato(nombre)
                        score = principal.get('score')
                        puntuacion = f'{score:.1%}' if isinstance(score, (float, int)) else 'no disponible'
                        comunes = (principal.get('species') or {}).get('commonNames') or []
                        gbif_id = (principal.get('gbif') or {}).get('id')

                        if clase:
                            st.success(f'Pl@ntNet propone una especie incluida en el proyecto: **{NOMBRES[clase]}** (*{clase}*).')
                            st.write(f'Puntuación devuelta por Pl@ntNet: **{puntuacion}**. Es una candidata automática, no una confirmación botánica.')
                            if comunes:
                                st.caption('Nombre(s) común(es) devuelto(s): ' + ', '.join(comunes[:3]))
                            if especie_local == clase:
                                st.info('El clasificador local y Pl@ntNet coinciden en esta clase.')
                            else:
                                st.warning(f'Los modelos discrepan: el clasificador local propone **{NOMBRES.get(especie_local, especie_local)}**; Pl@ntNet propone **{NOMBRES[clase]}**. Requiere revisión docente.')
                        else:
                            st.warning('La mejor candidata de Pl@ntNet **no pertenece a las cuatro clases del proyecto**. Se marca como fuera del catálogo del prototipo; sigue siendo una identificación automática por verificar.')
                            st.subheader(f'Candidata externa: *{nombre or respuesta.get("bestMatch", "especie no especificada")}*')
                            st.write(f'Puntuación devuelta por Pl@ntNet: **{puntuacion}**.')
                            if comunes:
                                st.caption('Nombre(s) común(es) devuelto(s): ' + ', '.join(comunes[:3]))
                            st.caption(f'El clasificador local, limitado a las cuatro clases, habría propuesto {NOMBRES.get(especie_local, especie_local)}. Esa salida local no confirma que la planta pertenezca al catálogo.')

                        if gbif_id:
                            mostrar_referencia_gbif(principal)
                        mostrar_imagenes_de_referencia(principal)

                        if len(candidatos) > 1:
                            with st.expander('Otras candidatas devueltas por Pl@ntNet'):
                                for candidato in candidatos[1:]:
                                    nombre_alterno = especie_de_candidato(candidato, respuesta) or 'Especie no especificada'
                                    score_alterno = candidato.get('score')
                                    texto_score = f'{score_alterno:.1%}' if isinstance(score_alterno, (float, int)) else 'puntuación no disponible'
                                    st.write(f'- *{nombre_alterno}* — {texto_score}')
                                    mostrar_referencia_gbif(candidato)

                        restante = respuesta.get('remainingIdentificationRequests')
                        if isinstance(restante, int):
                            st.caption(f'Consultas restantes informadas por Pl@ntNet hoy: {restante}.')
                except requests.Timeout:
                    st.error('La consulta a Pl@ntNet agotó el tiempo de espera. No se presenta una identificación.')
                except requests.RequestException:
                    st.error('No se pudo conectar con Pl@ntNet. No se presenta una identificación.')
                except RuntimeError as error:
                    st.error(str(error))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f'No se pudo procesar la imagen. {error}')

with st.expander('Método y resultados del proyecto'):
    st.write('La muestra del proyecto reúne 77 fotografías: 47 de desarrollo, 16 de exploración y 14 observaciones dominicanas reservadas para revisión. La reserva local no incluye las cuatro clases y no valida la exactitud en el Cibao.')
    st.table({'Método': ['Regresión logística', 'Bosque aleatorio', 'MobileNetV3 + regresión logística'],
              'Exactitud media interna': ['30,1 %', '31,8 %', '46,9 %']})
    st.write('El clasificador local solo distingue las cuatro especies entrenadas. Pl@ntNet aporta una búsqueda taxonómica externa; sus candidatas también requieren revisión docente. La evaluación del aprendizaje de los estudiantes y la validación de campo permanecen pendientes.')
st.download_button('Descargar cuaderno', (ROOT / 'Plantas_forrajeras_Cibao.ipynb').read_bytes(),
                   file_name='Plantas_forrajeras_Cibao.ipynb')
st.download_button('Descargar documento técnico de avance', (ROOT / 'Documento_tecnico_avance.pdf').read_bytes(),
                   file_name='Documento_tecnico_avance.pdf', mime='application/pdf')
