FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "-r" requirements.txt gunicorn
EXPOSE 88
CMD ["gunicorn", "-b", "0.0.0.0:88", "app:app"]