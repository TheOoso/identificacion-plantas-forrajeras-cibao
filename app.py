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

st.set_page_config(page_title='Plantas forrajeras del Cibao', page_icon='🌿', layout='wide')
st.markdown('''
<style>

:root{
  color-scheme:dark;
  --noche:#06151A; --cian:#35E0C2; --lima:#B8F34A; --texto:#E6F4EE; --suave:#93B4A9;
  --vidrio:rgba(255,255,255,.06); --borde:rgba(255,255,255,.13);
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
html{scroll-padding-top:env(safe-area-inset-top,0px);}
*{box-sizing:border-box;}
[data-testid="stAppViewContainer"]{margin:0;background:var(--noche);color:var(--texto);font-family:Manrope,system-ui,sans-serif;line-height:1.55;overflow-x:hidden;}
body::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(60% 50% at 12% 0%,rgba(53,224,194,.26),transparent 62%),
    radial-gradient(45% 40% at 95% 12%,rgba(184,243,74,.16),transparent 62%),
    radial-gradient(50% 40% at 50% 110%,rgba(53,224,194,.14),transparent 60%);
}
[data-testid="stHeader"]{background:transparent;} [data-testid="stMain"]>div{padding-top:.3rem;} .block-container{position:relative;z-index:1;max-width:1000px;padding:1rem 1.2rem 3.5rem;}
.vidrio{background:linear-gradient(145deg,rgba(255,255,255,.09),rgba(255,255,255,.03)); border:1px solid var(--borde); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);}

/* Hero */
.hero{position:relative; overflow:hidden; display:grid; grid-template-columns:1.15fr .85fr; align-items:center; gap:1rem; min-height:380px; border-radius:30px; padding:2.6rem 2.4rem; margin-bottom:1.4rem;}
.suelo{position:absolute; left:-25%; right:-25%; bottom:-34%; height:75%; transform:perspective(420px) rotateX(63deg); transform-origin:50% 100%;
  background:repeating-linear-gradient(90deg,rgba(53,224,194,.4) 0 1px,transparent 1px 52px),repeating-linear-gradient(0deg,rgba(53,224,194,.25) 0 1px,transparent 1px 52px);
  -webkit-mask-image:linear-gradient(to top,#000,transparent 88%); mask-image:linear-gradient(to top,#000,transparent 88%);
  animation:avanza 5s linear infinite;}
@keyframes avanza{to{background-position:0 52px,0 52px;}}
.hero .txt{position:relative; z-index:2;}
.hero h1{font-family:'Sora',sans-serif; font-weight:800; font-size:clamp(2rem,5vw,3.2rem); line-height:1.06; letter-spacing:-.025em; margin:0 0 .9rem;}
.hero p{color:#C3DBD2; max-width:46ch; margin:0 0 1.3rem; font-size:1.06rem;}
.chips{display:flex; flex-wrap:wrap; gap:.45rem;}
.chip{padding:.3rem .8rem; border-radius:999px; font-size:.88rem; font-weight:500; background:rgba(53,224,194,.1); border:1px solid rgba(53,224,194,.4);}

/* Escena 3D */
.escena{position:relative; z-index:2; display:grid; place-items:center; perspective:850px; height:280px;}
.orb{position:relative; width:220px; height:220px; transform-style:preserve-3d; animation:gira 16s linear infinite;}
@keyframes gira{from{transform:rotateX(-20deg) rotateY(0);}to{transform:rotateX(-20deg) rotateY(360deg);}}
.anillo{position:absolute; inset:0; border-radius:50%; border:1.5px solid rgba(184,243,74,.7); box-shadow:0 0 22px rgba(184,243,74,.28),inset 0 0 22px rgba(184,243,74,.14);}
.a2{transform:rotateX(72deg); border-color:rgba(53,224,194,.85); box-shadow:0 0 22px rgba(53,224,194,.3),inset 0 0 22px rgba(53,224,194,.15);}
.a3{transform:rotateY(72deg); inset:14px;}
.sat{position:absolute; top:50%; left:50%; width:14px; height:14px; margin:-7px; border-radius:50%; background:radial-gradient(circle at 30% 30%,#fff,var(--lima) 45%,#4a8f10); box-shadow:0 0 14px var(--lima); animation:orbita 6s linear infinite;}
.sat.s2{animation-duration:9s; animation-direction:reverse; background:radial-gradient(circle at 30% 30%,#fff,var(--cian) 45%,#0b6c5d); box-shadow:0 0 14px var(--cian);}
@keyframes orbita{from{transform:rotateY(0) translateX(110px);}to{transform:rotateY(360deg) translateX(110px);}}
.hoja{position:absolute; width:120px; height:120px; animation:flota 4.5s ease-in-out infinite;}
@keyframes flota{50%{transform:translateY(-12px);}}

/* Extrusión 3D de los iconos */
.ext{filter:drop-shadow(1px 1px 0 #0b6b55) drop-shadow(2px 2px 0 #0a5946) drop-shadow(3px 3px 0 #08463a) drop-shadow(4px 4px 0 #06382f) drop-shadow(0 16px 16px rgba(0,0,0,.55));}

/* Clases locales */
h2{font-family:'Sora',sans-serif; font-size:1.15rem; margin:1.8rem 0 .8rem; font-weight:600;}
.grid{display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin-bottom:1.4rem;}
.tarjeta{perspective:720px;}
.cara{position:relative; transform-style:preserve-3d; transform:rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg)); transition:transform .18s ease-out; border-radius:22px; padding:1.3rem .8rem 1.1rem; text-align:center;}
.cara::after{content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none; background:radial-gradient(circle at var(--mx,50%) var(--my,0%),rgba(184,243,74,.2),transparent 55%);}
.ico{display:block; width:78px; height:78px; margin:0 auto .7rem; transform:translateZ(48px); animation:flota 4s ease-in-out infinite; animation-delay:var(--d,0s);}
.cara b{display:block; font-family:'Sora',sans-serif; font-weight:600; font-size:.95rem; transform:translateZ(22px);}

/* Aviso */
.aviso{display:flex; gap:.9rem; align-items:flex-start; border-radius:18px; padding:1rem 1.2rem; margin-bottom:1rem; border-left:4px solid var(--lima);}
.pulso{flex:none; width:10px; height:10px; margin-top:.5rem; border-radius:50%; background:var(--lima); box-shadow:0 0 0 0 rgba(184,243,74,.6); animation:pulsa 2s infinite;}
@keyframes pulsa{70%{box-shadow:0 0 0 12px rgba(184,243,74,0);}100%{box-shadow:0 0 0 0 rgba(184,243,74,0);}}
.nota{color:var(--suave); font-size:.9rem; margin:0 0 1.6rem;}

/* Carga: escáner */
.lbl{font-weight:600; margin-bottom:.5rem;}
.drop{--c:var(--cian); position:relative; overflow:hidden; display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap; border-radius:14px; padding:2.4rem 1.8rem; margin-bottom:1rem;
  background:
    linear-gradient(var(--c),var(--c)) left 10px top 10px/30px 2px no-repeat,linear-gradient(var(--c),var(--c)) left 10px top 10px/2px 30px no-repeat,
    linear-gradient(var(--c),var(--c)) right 10px top 10px/30px 2px no-repeat,linear-gradient(var(--c),var(--c)) right 10px top 10px/2px 30px no-repeat,
    linear-gradient(var(--c),var(--c)) left 10px bottom 10px/30px 2px no-repeat,linear-gradient(var(--c),var(--c)) left 10px bottom 10px/2px 30px no-repeat,
    linear-gradient(var(--c),var(--c)) right 10px bottom 10px/30px 2px no-repeat,linear-gradient(var(--c),var(--c)) right 10px bottom 10px/2px 30px no-repeat,
    rgba(255,255,255,.04);
  border:1px dashed rgba(53,224,194,.3);}
.drop::before{content:""; position:absolute; left:12px; right:12px; top:12%; height:2px; background:linear-gradient(90deg,transparent,var(--cian),transparent); box-shadow:0 0 14px var(--cian); animation:barre 3.4s ease-in-out infinite alternate;}
@keyframes barre{to{top:88%;}}
.drop>*{position:relative; z-index:1;}
.cubo-w{perspective:300px; width:36px; height:36px;}
.cubo{position:relative; width:36px; height:36px; transform-style:preserve-3d; animation:gira2 9s linear infinite;}
.cubo i{position:absolute; inset:0; border:1.5px solid var(--cian); background:rgba(53,224,194,.1);}
.cubo i:nth-child(1){transform:translateZ(18px);} .cubo i:nth-child(2){transform:rotateY(90deg) translateZ(18px);}
.cubo i:nth-child(3){transform:rotateY(180deg) translateZ(18px);} .cubo i:nth-child(4){transform:rotateY(-90deg) translateZ(18px);}
.cubo i:nth-child(5){transform:rotateX(90deg) translateZ(18px);} .cubo i:nth-child(6){transform:rotateX(-90deg) translateZ(18px);}
@keyframes gira2{from{transform:rotateX(-25deg) rotateY(0);}to{transform:rotateX(-25deg) rotateY(360deg);}}
.subir{display:inline-flex; align-items:center; gap:.5rem; border:0; border-radius:12px; padding:.7rem 1.3rem; cursor:pointer; font:600 1rem 'Manrope',sans-serif; color:#04231A; background:linear-gradient(120deg,var(--lima),var(--cian)); box-shadow:0 8px 24px -6px rgba(53,224,194,.6); transition:transform .15s,box-shadow .15s;}
.subir:hover{transform:translateY(-2px); box-shadow:0 12px 30px -6px rgba(184,243,74,.7);}
.drop span.fmt{color:var(--suave); font-size:.9rem;}

/* Expander y botones */
details.exp{border-radius:16px; padding:.9rem 1.2rem; margin-bottom:1rem;}
details.exp summary{cursor:pointer; font-weight:600; list-style:none;}
details.exp summary::before{content:"›"; display:inline-block; margin-right:.6rem; color:var(--cian); transition:transform .2s;}
details.exp[open] summary::before{transform:rotate(90deg);}
details.exp p{color:var(--suave); margin:.7rem 0 0;}
.btns{display:flex; flex-wrap:wrap; gap:.7rem;}
.btn{display:inline-flex; align-items:center; gap:.5rem; padding:.6rem 1.15rem; border-radius:12px; cursor:pointer; color:var(--texto); font:600 .95rem 'Manrope',sans-serif; background:var(--vidrio); border:1px solid rgba(53,224,194,.5); transition:background-color .15s,box-shadow .15s;}
.btn:hover{background:rgba(53,224,194,.16); box-shadow:0 0 20px rgba(53,224,194,.3);}
button:focus-visible,summary:focus-visible{outline:3px solid var(--lima); outline-offset:3px;}

@media (max-width:720px){
  .hero{grid-template-columns:1fr; padding:1.8rem 1.3rem;} .escena{height:240px;}
  .grid{grid-template-columns:repeat(2,1fr);}
}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}

/* Widgets Streamlit con la paleta y los paneles del diseño de referencia */
[data-testid="stFileUploader"] section{position:relative;overflow:hidden;border:1px dashed rgba(53,224,194,.45)!important;border-radius:14px!important;padding:1.6rem 1.25rem!important;background:linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) left 10px top 10px/28px 2px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) left 10px top 10px/2px 28px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) right 10px top 10px/28px 2px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) right 10px top 10px/2px 28px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) left 10px bottom 10px/28px 2px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) left 10px bottom 10px/2px 28px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) right 10px bottom 10px/28px 2px no-repeat,linear-gradient(rgba(53,224,194,.72),rgba(53,224,194,.72)) right 10px bottom 10px/2px 28px no-repeat,rgba(255,255,255,.035)!important;box-shadow:inset 0 0 28px rgba(53,224,194,.035)}
[data-testid="stFileUploader"] section:before{content:"";position:absolute;z-index:0;left:12px;right:12px;top:12%;height:2px;background:linear-gradient(90deg,transparent,var(--cian),transparent);box-shadow:0 0 14px var(--cian);opacity:.72;animation:barre 3.4s ease-in-out infinite alternate}
[data-testid="stFileUploader"] section>*{position:relative;z-index:1}[data-testid="stFileUploader"] button{border:0!important;border-radius:12px!important;padding:.7rem 1.3rem!important;color:#04231a!important;background:linear-gradient(120deg,var(--lima),var(--cian))!important;font:600 1rem Manrope,system-ui,sans-serif!important;box-shadow:0 8px 24px -6px rgba(53,224,194,.6)}
[data-testid="stFileUploaderDropzoneInstructions"]{color:var(--texto)!important}[data-testid="stFileUploaderDropzoneInstructions"] small{color:var(--suave)!important}
[data-testid="stImage"] img{border-radius:16px;border:1px solid rgba(53,224,194,.28);box-shadow:0 14px 34px rgba(0,0,0,.4)}[data-testid="stImage"] p{color:var(--suave)!important}
[data-testid="stButton"] button[kind="primary"]{border:0!important;border-radius:12px!important;color:#04231a!important;background:linear-gradient(120deg,var(--lima),var(--cian))!important;font:700 1rem Manrope,sans-serif!important;box-shadow:0 8px 24px -6px rgba(53,224,194,.6)}
[data-testid="stAlert"]{border:1px solid var(--borde)!important;border-radius:16px!important;background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025))!important;color:var(--texto)!important;backdrop-filter:blur(10px)}[data-testid="stAlert"] p{color:var(--texto)!important}
[data-testid="stExpander"]{border:1px solid var(--borde)!important;border-radius:16px!important;background:linear-gradient(145deg,rgba(255,255,255,.06),rgba(255,255,255,.025))!important;backdrop-filter:blur(10px);margin-bottom:1rem}[data-testid="stExpander"] summary{color:var(--texto)!important;font-weight:700!important}[data-testid="stExpander"] svg{color:var(--cian)!important}
[data-testid="stTable"]{border:1px solid var(--borde);border-radius:14px;overflow:hidden}[data-testid="stTable"] th{background:rgba(53,224,194,.14)!important;color:#d8fff4!important}[data-testid="stTable"] td{background:rgba(255,255,255,.035)!important;color:var(--texto)!important}
[data-testid="stDownloadButton"] button,[data-testid="stLinkButton"] a{width:100%;min-height:46px;border:1px solid rgba(53,224,194,.5)!important;border-radius:12px!important;color:var(--texto)!important;background:var(--vidrio)!important;font:600 .95rem Manrope,sans-serif!important;transition:background-color .15s,box-shadow .15s}[data-testid="stDownloadButton"] button:hover,[data-testid="stLinkButton"] a:hover{background:rgba(53,224,194,.16)!important;box-shadow:0 0 20px rgba(53,224,194,.3)}
[data-testid="stCaptionContainer"]{color:var(--suave)!important}[data-testid="stCode"]{border:1px solid var(--borde);border-radius:12px;background:rgba(255,255,255,.04)}h2{font-family:Sora,Manrope,sans-serif!important;color:var(--texto)!important;font-size:1.15rem!important;margin:1.8rem 0 .8rem!important}
@media(max-width:720px){.block-container{padding:.65rem .85rem 2.5rem}.hero{grid-template-columns:1fr!important;padding:1.8rem 1.3rem!important}.escena{height:230px!important}.grid{grid-template-columns:repeat(2,1fr)!important}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}

</style>
''', unsafe_allow_html=True)
st.markdown('''
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#D9FF7A"/><stop offset=".55" stop-color="#7BEA8A"/><stop offset="1" stop-color="#35E0C2"/></linearGradient></defs></svg>
<section class="hero vidrio">
    <div class="suelo" aria-hidden="true"></div>
    <div class="txt">
      <h1>Plantas forrajeras del Cibao</h1>
      <p>Identificación asistida por imágenes para las clases de Pastos y Forrajes del Politécnico José Antonio Castillo.</p>
      <div class="chips"><span class="chip">Hierba de Guinea</span><span class="chip">Grama de Bermuda</span><span class="chip">Piñón cubano</span><span class="chip">Pasto bahía</span></div>
    </div>
    <div class="escena" aria-hidden="true">
      <div class="orb"><div class="anillo"></div><div class="anillo a2"></div><div class="anillo a3"></div><div class="sat"></div><div class="sat s2"></div></div>
      <svg class="hoja ext" viewBox="0 0 64 64"><path fill="url(#g)" d="M32 4C52 12 58 34 46 50 40 59 24 59 17 52 6 38 12 15 32 4Z"/><path d="M32 12V56M32 30L22 22M32 38L44 28M32 46L22 40" stroke="#0b6b55" stroke-width="2.2" stroke-linecap="round" fill="none"/></svg>
    </div>
  </section>
''', unsafe_allow_html=True)
st.markdown('<div class="aviso vidrio"><span class="pulso"></span><div>Prototipo experimental. La validación interna del clasificador local obtuvo <strong>46,9 %</strong> de exactitud media. Toda identificación automática es una candidata y requiere revisión docente.</div></div>', unsafe_allow_html=True)
st.markdown('<h2>Clases locales</h2><div class="grid">\n    <div class="tarjeta"><div class="cara vidrio"><svg class="ico ext" viewBox="0 0 64 64" style="--d:0s"><g fill="url(#g)"><path d="M32 58C28 38 20 24 10 14 24 18 33 34 32 58Z"/><path d="M32 58C32 36 36 20 44 6 46 24 38 42 32 58Z"/><path d="M32 58C36 42 46 32 56 28 50 42 42 52 32 58Z"/><path d="M32 58C28 44 18 36 8 34 14 46 22 54 32 58Z"/></g></svg><b>Hierba de Guinea</b></div></div>\n    <div class="tarjeta"><div class="cara vidrio"><svg class="ico ext" viewBox="0 0 64 64" style="--d:.5s"><g fill="url(#g)"><path d="M10 48L13 28 19 47Z"/><path d="M25 49L30 22 35 49Z"/><path d="M40 48L46 30 49 47Z"/><path d="M52 45L58 28 59 43Z"/></g><path d="M3 50C20 44 40 54 61 44" stroke="url(#g)" stroke-width="3.5" stroke-linecap="round" fill="none"/></svg><b>Grama de Bermuda</b></div></div>\n    <div class="tarjeta"><div class="cara vidrio"><svg class="ico ext" viewBox="0 0 64 64" style="--d:1s"><path d="M32 60V6" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/><g fill="url(#g)"><ellipse cx="21" cy="46" rx="10" ry="4.6" transform="rotate(-22 21 46)"/><ellipse cx="43" cy="46" rx="10" ry="4.6" transform="rotate(22 43 46)"/><ellipse cx="21" cy="31" rx="10" ry="4.6" transform="rotate(-22 21 31)"/><ellipse cx="43" cy="31" rx="10" ry="4.6" transform="rotate(22 43 31)"/><ellipse cx="22" cy="17" rx="9" ry="4.2" transform="rotate(-22 22 17)"/><ellipse cx="42" cy="17" rx="9" ry="4.2" transform="rotate(22 42 17)"/><ellipse cx="32" cy="7" rx="4.2" ry="6"/></g></svg><b>Piñón cubano</b></div></div>\n    <div class="tarjeta"><div class="cara vidrio"><svg class="ico ext" viewBox="0 0 64 64" style="--d:1.5s"><g stroke="url(#g)" fill="none" stroke-linecap="round"><path d="M32 60V30" stroke-width="3.5"/><path d="M32 30C26 22 20 14 15 4" stroke-width="5" stroke-dasharray="7 3.5"/><path d="M32 30C38 22 44 14 49 4" stroke-width="5" stroke-dasharray="7 3.5"/></g></svg><b>Pasto bahía</b></div></div>\n  </div>', unsafe_allow_html=True)
st.markdown('<p class="nota">Al pulsar «Analizar fotografía», la imagen se enviará a Pl@ntNet para buscar una candidata taxonómica y compararla con las cuatro clases. La imagen se convierte a JPEG y se eliminan sus metadatos EXIF antes del envío. No uses una imagen que no tengas autorización para compartir.</p>', unsafe_allow_html=True)
st.markdown('<div class="lbl">Carga una fotografía de una sola planta</div>', unsafe_allow_html=True)
foto = st.file_uploader('Carga una fotografía de una sola planta', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed', max_upload_size=10)


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
descarga_cuaderno, descarga_documento = st.columns(2)
with descarga_cuaderno:
    st.download_button('Descargar cuaderno', (ROOT / 'Plantas_forrajeras_Cibao.ipynb').read_bytes(),
                       file_name='Plantas_forrajeras_Cibao.ipynb', use_container_width=True)
with descarga_documento:
    st.download_button('Descargar documento técnico de avance', (ROOT / 'Documento_tecnico_avance.pdf').read_bytes(),
                       file_name='Documento_tecnico_avance.pdf', mime='application/pdf', use_container_width=True)

st.link_button('Ver repositorio del proyecto en GitHub', 'https://github.com/TheOoso/identificacion-plantas-forrajeras-cibao')
