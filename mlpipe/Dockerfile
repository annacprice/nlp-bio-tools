FROM python:2.7-slim

WORKDIR /app

COPY mlpipe.py /app

RUN apt-get update && \
apt-get install -y pkg-config libopenblas-dev liblapack-dev build-essential \
gfortran python-dev libfreetype6-dev libjpeg-dev libpng-dev libhdf5-dev liblzo2-dev libbz2-dev && \
pip install --no-cache-dir numpy==1.8.2 && \
pip install --no-cache-dir scipy==0.13.3 && \
pip install --no-cache-dir scikit-learn==0.19.1 && \
pip install --no-cache-dir python-dateutil==2.8.0 && \
pip install --no-cache-dir tornado==5.1.1 && \
pip install --no-cache-dir pyparsing==2.4.2 && \
pip install --no-cache-dir matplotlib==1.3.1


ENTRYPOINT ["python", "mlpipe.py"]
