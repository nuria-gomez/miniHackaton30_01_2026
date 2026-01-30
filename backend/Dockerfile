FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación backend
COPY backend/*.py ./

# Copiar frontend dist (construido previamente y copiado a backend/dist)
COPY backend/dist ./dist

# Crear directorio para uploads con permisos correctos
RUN mkdir -p /app/uploads && chmod 755 /app/uploads

# Verificar que dist existe (opcional, para debugging)
RUN ls -la /app/dist/ || echo "WARNING: dist directory not found"

# Exponer puerto
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
