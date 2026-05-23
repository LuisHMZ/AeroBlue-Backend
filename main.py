from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

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