# # Use an official Python runtime as the base image
# FROM python:3.11

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# # Set the working directory in the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt

# # Copy the Django project
# COPY . .

# # Create the static directory
# RUN mkdir -p staticfiles 

# # Ensure Django settings are properly set
# ENV DJANGO_SETTINGS_MODULE=anther_core.settings

# # Collect static files
# RUN python manage.py collectstatic --noinput -v 2

# # Run gunicorn
# CMD gunicorn anther_core.wsgi:application --bind 0.0.0.0:$PORT --log-level debug --access-logfile - --workers=2 --worker-class=gthread --threads=4


# Stage 1: Build dependencies
FROM python:3.11-slim-buster as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim-buster

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY . .

RUN if [ "$GOOGLE_CLOUD_PROJECT" ]; then python manage.py collectstatic --verbosity 3 --noinput; fi

CMD gunicorn anther_core.wsgi:application --bind 0.0.0.0:$PORT