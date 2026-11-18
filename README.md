# Plantas forrajeras del Cibao

Prototipo educativo del Ing. Edward Caraballo para las clases de Pastos y Forrajes del Politécnico José Antonio Castillo. Proyecto final IAE-2501, Maestría en Inteligencia Artificial, Educación e Investigación.

## Ejecución

Requiere Python 3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

El repositorio incluye los pesos entrenados. Para reproducir el entrenamiento:

```bash
python entrenar.py
```

La primera ejecución descarga los pesos oficiales de MobileNetV3 Small.

## Consulta de especies fuera del catálogo

Al pulsar «Analizar fotografía», la aplicación consulta Pl@ntNet para comparar su candidata con las cuatro clases locales. La imagen se convierte a JPEG y se eliminan sus metadatos EXIF antes de enviarla. La app no presenta como identificación la salida forzada del clasificador local cuando no está disponible la consulta externa.

La consulta requiere una clave API privada válida de Pl@ntNet. En Streamlit Cloud, configúrala en **App settings → Secrets**:

```toml
PLANTNET_API_KEY = "tu_clave_privada"
```

En ejecución local puede definirse la variable de entorno `PLANTNET_API_KEY`. No guardes la clave en el código, este README ni GitHub. La aplicación presenta el resultado externo como candidato por verificar, muestra referencias taxonómicas de GBIF cuando están disponibles y conserva la propuesta local solo como comparación. Si la clave no es válida o el servicio no devuelve candidata, la consulta externa puede fallar; la propuesta automática no constituye confirmación botánica.

La fotografía se envía a Pl@ntNet al solicitar el análisis. Según las condiciones del servicio, las imágenes se procesan para la identificación; el historial de consultas puede conservar fecha, hora y URL(s) de imagen, pero no los archivos de imagen.

## Datos y alcance

El cuaderno incorpora 77 imágenes reducidas de observaciones de iNaturalist con grado de investigación y licencias CC BY, CC BY-SA o CC0. Las imágenes transformadas conservan las condiciones de sus respectivas licencias. Los créditos y enlaces se encuentran en el cuaderno y en `creditos_imagenes.csv`.

Cuatro clases: *Gliricidia sepium*, *Megathyrsus maximus*, *Cynodon dactylon* y *Paspalum notatum*. La yerbita de leche quedó excluida por decisión del autor.

La selección es de conveniencia. Los nombres locales y las localidades reportadas requieren revisión botánica y geográfica para uso en campo. Las particiones internas agrupan por firma de autor; la independencia de ejemplares no está plenamente verificada.

## Resultados

La exactitud media interna de MobileNetV3 y regresión logística fue 46,9 %, con valores por partición de 18,8 %, 53,3 % y 68,8 %. La media F1 macro fue 0,478. La muestra de desarrollo comprende 47 imágenes. Se reservaron 16 para exploración y 14 observaciones dominicanas para validación local pendiente.

El clasificador local solo propone una de cuatro especies y no puede detectar por sí solo plantas fuera de esas clases. La consulta de Pl@ntNet ofrece una candidata externa, pero no constituye validación botánica. La revisión docente forma parte de la interpretación. El proyecto requiere evaluación local y una muestra más representativa. El uso forrajero debe verificarse con bibliografía técnica y revisión especializada.

## Entregables y enlaces

- **Documento técnico final:** [Documento_tecnico_final.pdf](Documento_tecnico_final.pdf)
- **Cuaderno Jupyter:** [Plantas_forrajeras_Cibao.ipynb](Plantas_forrajeras_Cibao.ipynb)
- **Aplicación web publicada:** [Abrir prototipo en Streamlit](https://identificacion-plantas-forrajeras-cibao-kyhuknyazfrn38krhh87ny.streamlit.app/)
- **Video de presentación:** [video_explicativo.mp4](video_explicativo.mp4)
- Código: `app.py`, `entrenar.py` y `requirements.txt`; modelos en `modelos/`; atribuciones en `creditos_imagenes.csv`.

La consulta de Pl@ntNet depende de una clave API válida configurada en Streamlit Secrets. Las métricas son resultados internos de esta muestra: no demuestran exactitud en el Cibao, impacto educativo ni aptitud de una planta para alimentación animal.
