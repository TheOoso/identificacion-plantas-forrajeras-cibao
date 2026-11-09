"""Reproduce el experimento guardado y exporta el prototipo."""
from pathlib import Path
import json
import joblib
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

root = Path(__file__).resolve().parent
cuaderno = json.loads((root / 'Plantas_forrajeras_Cibao.ipynb').read_text())
contexto = {}
for indice in (15, 17, 21, 24):
    fuente = cuaderno['cells'][indice]['source']
    exec(''.join(fuente) if isinstance(fuente, list) else fuente, contexto)
modelo = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
modelo.fit(contexto['X_transfer'], contexto['y_desarrollo'])
(root / 'modelos').mkdir(exist_ok=True)
joblib.dump(modelo, root / 'modelos' / 'clasificador.joblib')
torch.save(contexto['extractor'].state_dict(), root / 'modelos' / 'extractor.pt')
print('Modelo exportado. Entrenamiento con las 47 imágenes de desarrollo.')
