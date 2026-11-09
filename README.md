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

## Datos y alcance
El cuaderno incorpora 77 imágenes reducidas de observaciones de iNaturalist con grado de investigación, atribución y licencias CC BY, CC BY-SA o CC0. Las imágenes transformadas conservan las condiciones de sus respectivas licencias. Sus créditos y enlaces se encuentran en la celda de muestra incorporada.

Cuatro especies: Gliricidia sepium, Megathyrsus maximus, Cynodon dactylon y Paspalum notatum. La yerbita de leche quedó excluida por decisión del autor.

La selección es de conveniencia. Los nombres locales y las localidades reportadas requieren revisión botánica y geográfica para uso en campo. Las tres particiones internas agrupan por firma de autor. La independencia de ejemplares no está plenamente verificada.

## Resultados
La exactitud media interna de MobileNetV3 y regresión logística fue 46,9 %, con valores por partición de 18,8 %, 53,3 % y 68,8 %. La media F1 macro fue 0,478. La muestra de desarrollo comprende 47 imágenes. Se reservaron 16 para exploración y 14 observaciones dominicanas para validación local pendiente.

La aplicación propone una de cuatro especies. La revisión docente forma parte de la interpretación. El proyecto todavía requiere evaluación local y una muestra más representativa. El uso forrajero debe consultarse en bibliografía técnica.

## Estado de entrega
Cuaderno, pesos entrenados, aplicación, créditos de imágenes y documento técnico de avance preparados. El documento técnico se organizó con los apartados de diseño de proyecto de la guía del docente. Carga del modelo e inferencia comprobadas con cuatro imágenes; esta comprobación no mide exactitud. Video y enlaces de publicación pendientes de completar. La justificación botánica y forrajera requiere referencias técnicas verificadas antes de cerrar la versión final.
