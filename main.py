from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import math
import random

# 1. Inicializamos la aplicación
app = FastAPI(
    title="AeroBlue API",
    description="Motor predictivo de calidad del aire para IPN Zacatenco",
    version="1.0.0"
)

modelo = None
columnas_requeridas = None

# 2. El "Cadenero" (Definimos qué datos debe recibir el servidor)
# Estas columnas son exactamente las que extrajimos de la SEDEMA
class DatosAtmosfericos(BaseModel):
    NO2: float
    O3: float
    PM10: float
    PMCO: float
    RH: float     # Humedad Relativa
    TMP: float    # Temperatura
    WDR: float    # Dirección del viento
    WSP: float    # Velocidad del viento

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

# 4. Ruta de prueba
@app.get("/")
def ruta_raiz():
    return {"estado": "En linea, esperando datos..."}

# 5. LA RUTA MAESTRA: Donde ocurre la magia
@app.post("/predecir")
def predecir_pm25(datos: DatosAtmosfericos):
    # a) Convertimos el JSON que llega de internet a un formato que Pandas entienda
    df_entrada = pd.DataFrame([datos.dict()])
    
    # b) Nos aseguramos de que las columnas estén en el orden exacto que requiere el modelo
    df_entrada = df_entrada[columnas_requeridas]
    
    # c) Hacemos la predicción matemática
    prediccion = modelo.predict(df_entrada)
    resultado = prediccion[0]
    
    # d) Lógica de negocio (AeroBlue): ¿Es peligroso?
    # La OMS dicta que un PM2.5 mayor a 45 en 24h es dañino. 
    alerta = False
    if resultado > 45.0:
        alerta = True
        
    # e) Devolvemos la respuesta a la app
    return {
        "pm25_calculado": round(resultado, 2),
        "alerta_contingencia": alerta,
        "zona": "GAM (Zacatenco)"
    }

# 6. RUTA PARA GRÁFICAS (EVOLUCIÓN Y PRONÓSTICO EXTENDIDO)
@app.post("/api/pronostico_graficas/")
def pronostico_graficas(datos: DatosAtmosfericos):
    global modelo, columnas_requeridas

    if modelo is None:
        return {"error": "El modelo de Machine Learning no está cargado."}

    escenarios_climaticos = []

    # === A. CREAR MATRIZ DE 24 HORAS ===
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

    # === B. CREAR MATRIZ DE 3 SEMANAS (21 DÍAS) ===
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

    # Convertimos a DataFrame de Pandas
    df_futuro = pd.DataFrame(escenarios_climaticos)

    # CRÍTICO: Ordenamos las columnas exactamente como lo exige tu archivo .pkl
    df_futuro = df_futuro[columnas_requeridas]

    # Predicción masiva superrápida (45 filas al mismo tiempo)
    predicciones = modelo.predict(df_futuro)
    resultados = [round(float(p), 1) for p in predicciones]

    return {
        "hoy": resultados[0:24],
        "mes": {
            "Semana 2 del mes": resultados[24:31],
            "Semana 3 del mes": resultados[31:38],
            "Semana 4 del mes": resultados[38:45],
        }
    }

# 7. NUEVA RUTA: GENERADOR DETERMINISTA DE HISTORIAL
@app.get("/api/historico/{fecha}")
def obtener_historico(fecha: str):
    """
    Endpoint para simular el historial de calidad del aire de un día específico.
    Utiliza la fecha como semilla para garantizar que siempre devuelva la misma
    curva al consultar el mismo día, imitando el comportamiento de una base de datos.
    """
    # La clave de la magia: Semilla basada en el texto de la fecha (Ej. "2026-06-03")
    random.seed(fecha)

    pm25_data = []
    pm10_data = []
    o3_data = []

    # Generamos un nivel de contaminación base para ese día (entre excelente y moderado)
    base_pm25 = random.uniform(8.0, 25.0)

    for hora in range(24):
        # Curva natural: Menos contaminación en madrugada, picos en el día
        variacion = math.sin((hora - 6) / 24.0 * math.pi) * 15.0
        ruido = random.uniform(-2.0, 2.0)
        
        # Calculamos el valor final asegurando que nunca sea negativo o irreal
        val_pm25 = max(5.0, base_pm25 + variacion + ruido)

        # Generamos los otros dos contaminantes proporcionalmente al PM2.5
        pm25_data.append(round(val_pm25, 1))
        pm10_data.append(round(val_pm25 * random.uniform(1.2, 1.8), 1))
        o3_data.append(round(val_pm25 * random.uniform(0.5, 0.9), 1))

    # Reseteamos la semilla aleatoria para no afectar otras funciones del servidor
    random.seed()

    # Estructura JSON para la pantalla de Análisis Histórico de Flutter
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