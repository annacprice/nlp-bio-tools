FROM python:2.7-slim

WORKDIR /app

COPY loadmodel.py /app 

RUN apt-get update && \
apt-get install -y pkg-config libopenblas-dev liblapack-dev build-essential gfortran python-dev libfreetype6-dev libjpeg-dev libhdf5-dev liblzo2-dev libbz2-dev && \ 
pip install --no-cache-dir numpy==1.8.2 && \
pip install --no-cache-dir scipy==0.13.3 && \
pip install --no-cache-dir scikit-learn==0.19.1


ENTRYPOINT ["python", "loadmodel.py"]
