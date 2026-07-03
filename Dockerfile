FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Assuming Honcho uses a Procfile in the root
# Koyeb standard is 8080. If honcho doesn't support env-based port
# you might need to wrap it.
CMD ["honcho", "start"]
