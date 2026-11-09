"""Aplicación educativa de clasificación visual."""
from pathlib import Path
from io import BytesIO
import joblib
import streamlit as st
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
NOMBRES = {'Cynodon dactylon':'Grama de Bermuda', 'Gliricidia sepium':'Piñón cubano',
           'Megathyrsus maximus':'Hierba de Guinea', 'Paspalum notatum':'Hierba bahía'}
st.set_page_config(page_title='Plantas forrajeras del Cibao', page_icon='🌿', layout='centered')
st.title('Plantas forrajeras del Cibao')
st.write('Identificación asistida por imágenes para las clases de Pastos y Forrajes del Politécnico José Antonio Castillo.')
st.info('Prototipo experimental. La validación interna alcanzó 46,9 % de exactitud media. Contrasta la propuesta con los rasgos de la planta y la revisión docente.')
st.write('Especies incluidas: hierba de Guinea, grama de Bermuda, piñón cubano y hierba bahía.')

@st.cache_resource
def cargar():
    torch.set_num_threads(2)
    extractor = mobilenet_v3_small(weights=None)
    extractor.classifier = torch.nn.Identity()
    extractor.load_state_dict(torch.load(ROOT/'modelos/extractor.pt', map_location='cpu', weights_only=True))
    extractor.eval()
    return extractor, joblib.load(ROOT/'modelos/clasificador.joblib'), MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()

foto = st.file_uploader('Carga una fotografía de la planta', type=['jpg','jpeg','png'])
if foto is not None:
    try:
        if foto.size > 10*1024*1024:
            raise ValueError('La imagen debe pesar como máximo 10 MB.')
        with Image.open(BytesIO(foto.getvalue())) as origen:
            imagen = ImageOps.exif_transpose(origen).convert('RGB')
        st.image(imagen, caption='Fotografía cargada', width=380)
        if st.button('Analizar fotografía', type='primary'):
            if not (ROOT/'modelos/clasificador.joblib').exists():
                st.error('Faltan los archivos entrenados. Ejecuta entrenar.py para preparar el prototipo.')
            else:
                extractor, clasificador, preparar = cargar()
                with torch.inference_mode():
                    vector = extractor(preparar(imagen).unsqueeze(0)).numpy()
                especie = clasificador.predict(vector)[0]
                st.subheader('Propuesta del modelo')
                st.write(f'**{NOMBRES[especie]}** — *{especie}*')
                st.caption('El modelo asigna una de sus cuatro categorías. Una planta ajena a esas categorías también recibirá una propuesta. La identificación necesita comprobación docente.')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f'No se pudo procesar la imagen. {error}')
with st.expander('Método y resultados'):
    st.write('77 fotografías con licencia de reutilización. 47 imágenes de desarrollo, 16 de prueba exploratoria y 14 observaciones dominicanas reservadas. La validación interna utiliza tres particiones por autor.')
    st.table({'Método':['Regresión logística','Bosque aleatorio','MobileNetV3 + regresión logística'],
              'Exactitud media':['30,1 %','31,8 %','46,9 %']})
    st.write('La evaluación del aprendizaje de los estudiantes y la validación de campo permanecen pendientes.')
st.download_button('Descargar cuaderno', (ROOT/'Plantas_forrajeras_Cibao.ipynb').read_bytes(),
                   file_name='Plantas_forrajeras_Cibao.ipynb')
st.download_button('Descargar documento técnico de avance', (ROOT/'Documento_tecnico_avance.pdf').read_bytes(),
                   file_name='Documento_tecnico_avance.pdf', mime='application/pdf')
