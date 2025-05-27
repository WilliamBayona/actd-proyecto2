# Usa una imagen oficial de Python
FROM python:3.10-slim

# Crea un directorio para la app
WORKDIR /app

# Copia tus archivos
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde corre Dash
EXPOSE 8050

# Comando para correr la app
CMD ["python", "dashboard.py"]