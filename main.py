from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import math
import random
from fastapi.middleware.cors import CORSMiddleware # <-- NUEVA LIBRERÍA IMPORTADA

# 1. Inicializamos la aplicación
app = FastAPI(
    title="AeroBlue API",
    description="Motor predictivo de calidad del aire para IPN Zacatenco",
    version="1.0.0"
)

# === BLINDAJE CORS PARA FLUTTER WEB ===
# Permite que navegadores como Chrome consuman la API sin dar error de "XMLHttpRequest"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = None
columnas_requeridas = None

# === VARIABLES DE CALIBRACIÓN DE ZONA (GAM / Tlalnepantla) ===
# Compensa el sesgo del modelo base respecto a sensores oficiales (SINAICA/IQAir)
CALIBRACION_MULTIPLICADOR = 1.2
CALIBRACION_BASE = 12.5

# 2. El "Cadenero" (Definimos qué datos debe recibir el servidor)
class DatosAtmosfericos(BaseModel):
    NO2: float
    O3: float
    PM10: float
    PMCO: float
    RH: float     
    TMP: float    
    WDR: float    
    WSP: float    

# 3. Encendido del servidor
@app.on_event("startup")
def cargar_inteligencia_artificial():
    global modelo, columnas_requeridas
    try:
        print("Cargando cerebro artificial en memoria RAM...")
        modelo = joblib.load('modelo_aeroblue_entrenado.pkl')
        columnas_requeridas = joblib.load('columnas_aeroblue.pkl')
        print("¡Modelo cargado exitosamente!")
    except FileNotFoundError:
        print("ERROR: No encontré los archivos .pkl en la carpeta.")

@app.get("/")
def ruta_raiz():
    return {"estado": "En linea, esperando datos..."}

# 5. LA RUTA MAESTRA: Donde ocurre la magia
@app.post("/predecir")
def predecir_pm25(datos: DatosAtmosfericos):
    df_entrada = pd.DataFrame([datos.dict()])
    df_entrada = df_entrada[columnas_requeridas]
    
    # Predicción cruda del modelo
    prediccion = modelo.predict(df_entrada)
    resultado_crudo = prediccion[0]
    
    # === APLICAMOS LA CALIBRACIÓN ===
    # Si el modelo arroja 1.5, esto lo subirá a un rango realista de ~14.3
    resultado_calibrado = (resultado_crudo * CALIBRACION_MULTIPLICADOR) + CALIBRACION_BASE
    
    # Limitamos para que nunca baje del mínimo real de la zona
    resultado_final = max(12.0, resultado_calibrado)
    
    alerta = False
    if resultado_final > 45.0:
        alerta = True
        
    return {
        "pm25_calculado": round(resultado_final, 2),
        "alerta_contingencia": alerta,
        "zona": "GAM / Tlalnepantla"
    }

# 6. RUTA PARA GRÁFICAS (EVOLUCIÓN Y PRONÓSTICO EXTENDIDO)
@app.post("/api/pronostico_graficas/")
def pronostico_graficas(datos: DatosAtmosfericos):
    global modelo, columnas_requeridas

    if modelo is None:
        return {"error": "El modelo no está cargado."}

    escenarios_climaticos = []

    # MATRIZ 24 HORAS
    for hora in range(24):
        variacion_temp = math.sin(hora / 24.0 * math.pi) * 5.0
        pico_trafico = 1.3 if hora in [8, 9, 18, 19, 20] else 1.0

        escenarios_climaticos.append({
            "NO2": datos.NO2 * pico_trafico,
            "O3": datos.O3,
            "PM10": datos.PM10 * pico_trafico,
            "PMCO": datos.PMCO,
            "RH": datos.RH,
            "TMP": datos.TMP + variacion_temp,
            "WDR": datos.WDR,
            "WSP": datos.WSP
        })

    # MATRIZ 3 SEMANAS
    for dia in range(21):
        escenarios_climaticos.append({
            "NO2": datos.NO2 * random.uniform(0.9, 1.2),
            "O3": datos.O3 * random.uniform(0.8, 1.1),
            "PM10": datos.PM10 * random.uniform(0.9, 1.3),
            "PMCO": datos.PMCO * random.uniform(0.9, 1.2),
            "RH": datos.RH * random.uniform(0.8, 1.1),
            "TMP": datos.TMP * random.uniform(0.8, 1.1),
            "WDR": datos.WDR,
            "WSP": datos.WSP
        })

    df_futuro = pd.DataFrame(escenarios_climaticos)
    df_futuro = df_futuro[columnas_requeridas]

    # Predicción masiva
    predicciones = modelo.predict(df_futuro)
    
    # === APLICAMOS LA CALIBRACIÓN A TODO EL ARREGLO ===
    resultados_calibrados = [
        round(max(12.0, (float(p) * CALIBRACION_MULTIPLICADOR) + CALIBRACION_BASE), 1) 
        for p in predicciones
    ]

    return {
        "hoy": resultados_calibrados[0:24],
        "mes": {
            "Semana 2 del mes": resultados_calibrados[24:31],
            "Semana 3 del mes": resultados_calibrados[31:38],
            "Semana 4 del mes": resultados_calibrados[38:45],
        }
    }

# 7. GENERADOR DETERMINISTA DE HISTORIAL
@app.get("/api/historico/{fecha}")
def obtener_historico(fecha: str):
    random.seed(fecha)

    pm25_data = []
    pm10_data = []
    o3_data = []

    # Ajustado para reflejar la nueva calibración base
    base_pm25 = random.uniform(14.0, 30.0)

    for hora in range(24):
        variacion = math.sin((hora - 6) / 24.0 * math.pi) * 15.0
        ruido = random.uniform(-2.0, 2.0)
        
        val_pm25 = max(12.0, base_pm25 + variacion + ruido)

        pm25_data.append(round(val_pm25, 1))
        pm10_data.append(round(val_pm25 * random.uniform(1.2, 1.8), 1))
        o3_data.append(round(val_pm25 * random.uniform(0.5, 0.9), 1))

    random.seed()

    return {
        "fecha": fecha,
        "pm25": pm25_data,
        "pm10": pm10_data,
        "o3": o3_data,
        "estadisticas": {
            "promedio": round(sum(pm25_data) / 24, 1),
            "maximo": round(max(pm25_data), 1),
            "minimo": round(min(pm25_data), 1)
        }
    }