FROM python:3.11-slim

# إعدادات بايثون
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TTS_FORCE_NONINTERACTIVE=1
ENV COQUI_TOS_AGREED=1
ENV TORCH_WEIGHTS_ONLY=0   

# تثبيت متطلبات الصوت و build tools
RUN apt-get update && apt-get install -y \
       default-jdk \
    unzip \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    build-essential \
    default-jre \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt
COPY download_xtts.py .
RUN python download_xtts.py

COPY . .

RUN mkdir -p uploads generated

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
