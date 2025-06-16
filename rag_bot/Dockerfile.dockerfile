FROM python:3.10

RUN apt-get update && \
    apt-get install -y libgl1 libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]