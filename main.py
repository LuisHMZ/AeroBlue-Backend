from fastapi import FastAPI, Depends 
from pydantic import BaseModel
import joblib
import pandas as pd
import math
import random
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session 
from sqlalchemy import text 
from database import get_db 

# 1. Inicializamos la aplicación
app = FastAPI(
    title="AeroBlue API",
    description="Motor predictivo de calidad del aire para IPN Zacatenco",
    version="1.0.0"
)

# === BLINDAJE CORS PARA FLUTTER WEB ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = None
columnas_requeridas = None

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

class ReporteSalud(BaseModel):
    sintoma: str
    pm25_al_momento: float
    latitud: float = None
    longitud: float = None

# === NUEVOS CADENEROS: Para los reportes de fallas de la App ===
class ReporteFallaCreate(BaseModel):
    descripcion: str
    dispositivo: str

class ActualizarEstadoRequest(BaseModel):
    estado: str

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
def predecir_pm25(datos: DatosAtmosfericos, db: Session = Depends(get_db)): 
    df_entrada = pd.DataFrame([datos.dict()])
    df_entrada = df_entrada[columnas_requeridas]
    
    # Predicción cruda del modelo
    prediccion = modelo.predict(df_entrada)
    resultado_crudo = prediccion[0]
    
    # === APLICAMOS LA CALIBRACIÓN ===
    resultado_calibrado = (resultado_crudo * CALIBRACION_MULTIPLICADOR) + CALIBRACION_BASE
    resultado_final = max(12.0, resultado_calibrado)
    
    alerta = False
    if resultado_final > 45.0:
        alerta = True

    # === GUARDADO SILENCIOSO EN NEON (DATA FLYWHEEL) ===
    try:
        sql = text("""
            INSERT INTO historial_predicciones 
            (pm25_predicho, temperatura, humedad) 
            VALUES (:pm25, :temp, :hum)
        """)
        db.execute(sql, {
            "pm25": round(resultado_final, 2),
            "temp": datos.TMP,
            "hum": datos.RH
        })
        db.commit()
    except Exception as e:
        print(f"Error guardando historial de BD: {e}")
        db.rollback() 
        
    return {
        "pm25_calculado": round(resultado_final, 2),
        "alerta_contingencia": alerta,
        "zona": "GAM / Tlalnepantla"
    }

# 6. RECOPILADOR DE SÍNTOMAS (CROWDSOURCING MÉDICO)
@app.post("/api/reporte-salud")
def guardar_reporte_salud(reporte: ReporteSalud, db: Session = Depends(get_db)):
    try:
        sql = text("""
            INSERT INTO reportes_salud_usuarios 
            (usuario_id, sintoma, pm25_al_momento, latitud, longitud) 
            VALUES (:uid, :sintoma, :pm25, :lat, :lon)
        """)
        db.execute(sql, {
            "uid": "usuario_local", 
            "sintoma": reporte.sintoma,
            "pm25": reporte.pm25_al_momento,
            "lat": reporte.latitud,
            "lon": reporte.longitud
        })
        db.commit()
        return {"estatus": "ok", "mensaje": "Síntomas guardados correctamente en la nube"}
    except Exception as e:
        db.rollback()
        return {"estatus": "error", "detalle": str(e)}

# === 7. NUEVO: RUTAS PARA GESTIÓN DE FALLAS Y SOPORTE ===

# A) Ruta para que Flutter ENVÍE el reporte a Neon
@app.post("/api/reportes", status_code=201)
def crear_reporte_falla(reporte: ReporteFallaCreate, db: Session = Depends(get_db)):
    try:
        sql = text("""
            INSERT INTO reportes_fallas (descripcion, dispositivo)
            VALUES (:desc, :disp)
        """)
        db.execute(sql, {
            "desc": reporte.descripcion,
            "disp": reporte.dispositivo
        })
        db.commit()
        return {"estatus": "ok", "mensaje": "Reporte de falla enviado exitosamente a la base de datos."}
    except Exception as e:
        db.rollback()
        return {"estatus": "error", "detalle": str(e)}

# B) Ruta para que tu FUTURA WEB LEA la lista de reportes
@app.get("/api/reportes")
def obtener_reportes(db: Session = Depends(get_db)):
    try:
        # Traemos todo ordenado del más nuevo al más viejo
        sql = text("SELECT id, fecha, descripcion, dispositivo, estado FROM reportes_fallas ORDER BY fecha DESC")
        resultados = db.execute(sql).fetchall()
        
        # Formateamos los resultados para devolver un JSON limpio
        reportes = []
        for fila in resultados:
            reportes.append({
                "id": fila[0],
                "fecha": fila[1],
                "descripcion": fila[2],
                "dispositivo": fila[3],
                "estado": fila[4]
            })
        return {"estatus": "ok", "datos": reportes}
    except Exception as e:
        return {"estatus": "error", "detalle": str(e)}

# C) Ruta para que tu FUTURA WEB ACTUALICE el estado (ej. "Resuelto")
@app.put("/api/reportes/{reporte_id}")
def actualizar_reporte(reporte_id: int, request: ActualizarEstadoRequest, db: Session = Depends(get_db)):
    try:
        sql = text("UPDATE reportes_fallas SET estado = :estado WHERE id = :id")
        db.execute(sql, {
            "estado": request.estado, 
            "id": reporte_id
        })
        db.commit()
        return {"estatus": "ok", "mensaje": f"Reporte {reporte_id} actualizado a estado: {request.estado}"}
    except Exception as e:
        db.rollback()
        return {"estatus": "error", "detalle": str(e)}

# 8. RUTA PARA GRÁFICAS (EVOLUCIÓN Y PRONÓSTICO EXTENDIDO)
@app.post("/api/pronostico_graficas/")
def pronostico_graficas(datos: DatosAtmosfericos):
    global modelo, columnas_requeridas

    if modelo is None:
        return {"error": "El modelo no está cargado."}

    escenarios_climaticos = []

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

    predicciones = modelo.predict(df_futuro)
    
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

# 9. GENERADOR DETERMINISTA DE HISTORIAL
@app.get("/api/historico/{fecha}")
def obtener_historico(fecha: str):
    random.seed(fecha)

    pm25_data = []
    pm10_data = []
    o3_data = []

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
    }#para el comit 
    #
    #
    #xd