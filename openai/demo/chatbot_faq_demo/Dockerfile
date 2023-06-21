FROM python:3.9

WORKDIR /app
COPY app .
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["streamlit", "run" , "app.py","--server.port","80"]
