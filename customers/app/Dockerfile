FROM python:3.11-slim


WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 80


CMD ["gunicorn", "app:app", "-b", "0.0.0.0:80"]