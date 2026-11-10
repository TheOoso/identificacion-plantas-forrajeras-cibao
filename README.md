# Plantas forrajeras del Cibao

Prototipo educativo del Ing. Edward Caraballo para Pastos y Forrajes en el Politécnico José Antonio Castillo. Proyecto IAE-2501, Maestría en Inteligencia Artificial, Educación e Investigación.

## Ejecución
Requiere Python 3.12. Instalar dependencias con:
    pip install -r requirements.txt
El paquete incluye los pesos entrenados. Para reproducir el entrenamiento:
    python entrenar.py
La primera ejecución descarga los pesos oficiales de MobileNetV3 Small.
Abrir la aplicación con:
    streamlit run app.py

## Consulta externa de especies fuera del catálogo

Al pulsar «Analizar fotografía», la aplicación consulta Pl@ntNet para comparar su candidata con las cuatro clases locales. La imagen se convierte a JPEG y se eliminan sus metadatos EXIF antes de enviarla. La app no presenta como identificación la salida forzada del clasificador local cuando no está disponible la consulta externa.

Se requiere una clave API privada de Pl@ntNet. Configúrala en Streamlit Cloud, en **App settings → Secrets**:

```toml
PLANTNET_API_KEY = "tu_clave_privada"
```

En ejecución local puede definirse la variable de entorno `PLANTNET_API_KEY`. No se debe guardar la clave en el código, en este README ni en GitHub. La aplicación presenta el resultado externo como candidato por verificar, muestra las referencias taxonómicas de GBIF cuando están disponibles y conserva la propuesta del modelo local solo como comparación.

La foto se envía a Pl@ntNet al solicitar el análisis. Según las condiciones del servicio, las imágenes se procesan para la identificación; el historial de consultas puede conservar fecha, hora y URL(s) de imagen, pero no los archivos de imagen.

## Datos y alcance
El cuaderno incorpora 77 imágenes reducidas de observaciones de iNaturalist con grado de investigación, atribución y licencias CC BY, CC BY-SA o CC0. Las imágenes transformadas conservan las condiciones de sus respectivas licencias. Sus créditos y enlaces se encuentran en la celda de muestra incorporada.

Cuatro especies: Gliricidia sepium, Megathyrsus maximus, Cynodon dactylon y Paspalum notatum. La yerbita de leche quedó excluida por decisión del autor.

La selección es de conveniencia. Los nombres locales y las localidades reportadas requieren revisión botánica y geográfica para uso en campo. Las tres particiones internas agrupan por firma de autor. La independencia de ejemplares no está plenamente verificada.

## Resultados
La exactitud media interna de MobileNetV3 y regresión logística fue 46,9 %, con valores por partición de 18,8 %, 53,3 % y 68,8 %. La media F1 macro fue 0,478. La muestra de desarrollo comprende 47 imágenes. Se reservaron 16 para exploración y 14 observaciones dominicanas para validación local pendiente.

El clasificador local solo fue entrenado para proponer una de cuatro especies y no puede detectar por sí solo plantas fuera de esas clases. La aplicación consulta Pl@ntNet al solicitar el análisis para obtener una candidata externa; esta consulta no constituye validación botánica. La revisión docente forma parte de la interpretación. El proyecto todavía requiere evaluación local y una muestra más representativa. El uso forrajero debe consultarse en bibliografía técnica.

## Estado de entrega
Cuaderno, pesos entrenados, aplicación, créditos de imágenes, documento técnico de avance y [video explicativo](video_explicativo.mp4) incluidos. El documento técnico se organizó con los apartados de diseño de proyecto de la guía del docente. Carga del modelo e inferencia comprobadas con cuatro imágenes; esta comprobación no mide exactitud. Falta registrar en este README los enlaces funcionales de la aplicación publicada. Se añadieron referencias técnicas para nomenclatura y descripciones generales; no verifican las etiquetas de las fotografías ni la presencia local en el Cibao.
