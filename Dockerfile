FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 7860

CMD bash -c "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & python frontend/app.py"