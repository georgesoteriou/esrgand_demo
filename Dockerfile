FROM pytorch/pytorch:latest
ENV PROD=true HOST=0.0.0.0 PORT=5000 TIMEOUT=60 GPU=1
WORKDIR /app
COPY server.py server.py
COPY evaluate.py evaluate.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY frontend/dist frontend/dist

CMD ["python", "server.py"]
