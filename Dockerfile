FROM python:3.8

WORKDIR /code

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY ./ ./

EXPOSE 8051

CMD ["python", "./app.py", "2021-02-06"]