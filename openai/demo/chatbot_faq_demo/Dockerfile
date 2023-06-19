FROM python:3.9

WORKDIR /app
COPY app .
COPY req req
RUN pip install -r req

CMD ["streamlit", "run" , "app.py","--server.port","80"]
