# Use an official Python runtime as the base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the Django project
COPY . .

# Create the static directory
RUN mkdir -p staticfiles 

# Ensure Django settings are properly set
ENV DJANGO_SETTINGS_MODULE=anther_core.settings

# Collect static files
RUN python manage.py collectstatic --noinput -v 2

# Run gunicorn
CMD gunicorn anther_core.wsgi:application --bind 0.0.0.0:$PORT --log-level debug --access-logfile - --workers=2 --worker-class=gthread --threads=4
