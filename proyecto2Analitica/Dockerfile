FROM python:3.9-slim

# 1. Ajustes de sistema
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# 2. Copiar requirements.txt
COPY requirements.txt .

# 3. Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# 4. Copiar la app y los datos/modelos
COPY dashboard.py .
COPY data2016_P1_limpio.csv .
COPY promedios_por_colegio.csv .
COPY scaler_balanceado.pkl .
COPY mejor_modelo_balanceado.h5 .
COPY mejor_modelo_balanceado.keras .

# 5. Variables y exposición de puerto
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
EXPOSE 8050

# 6. Arranque
CMD ["python", "dashboard.py"]