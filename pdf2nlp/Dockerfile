FROM python:2.7-alpine

WORKDIR /app

COPY pdf2nlp.py /app

RUN pip install --no-cache-dir pdfminer==20140328 && \
pip install --no-cache-dir nltk==3.2.5 && \
python -m nltk.downloader punkt stopwords


ENTRYPOINT ["python", "pdf2nlp.py"]
