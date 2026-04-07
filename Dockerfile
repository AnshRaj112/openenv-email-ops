FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 7860

CMD bash -c "export PORT=${PORT:-7860}; python -c 'from server.app import main; main()'"