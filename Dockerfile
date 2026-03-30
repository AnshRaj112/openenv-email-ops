FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 7860

CMD bash -c "export PORT=${PORT:-7860}; python -c 'from server.app import main; main()'"