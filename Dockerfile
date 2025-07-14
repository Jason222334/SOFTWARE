# Imagen base
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY app.py .
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/val/ ./data/val/
COPY sample_data/ ./sample_data/  

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
RUN ls -R ./data/val || echo "No se copiaron las imágenes correctamente"

# Puerto para Streamlit
EXPOSE 8501

# Ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
