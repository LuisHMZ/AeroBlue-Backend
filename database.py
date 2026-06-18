import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Cargamos las variables de entorno
load_dotenv()

# 2. Obtenemos la URL de conexión
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# 3. Creamos el motor de conexión CON LOS SEGUROS PARA NEON
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verfica si la conexión sigue viva antes de usarla
    pool_recycle=300     # Renueva la conexión automáticamente cada 5 minutos
)

# 4. Creamos la fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. Función para inyectar la base de datos en los endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()