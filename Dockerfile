# Usa una imagen base oficial de Python
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos (ajústalo si se llama diferente)
COPY requirements.txt .

# Instala dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto de tu aplicación
COPY app/ .

# Crea subdirectorios para imágenes
RUN mkdir -p imagenes labels resultados

# Expone el puerto de la API
EXPOSE 8000

# Comando para correr la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
