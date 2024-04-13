# Build the React frontend
FROM node:14.17.6-alpine as build-frontend
WORKDIR /static/react
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run react-build

# Build the Django backend
FROM python:3.9-slim-buster as build-backend
WORKDIR /Anther
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Final image
FROM python:3.9-slim-buster
WORKDIR /Anther
COPY --from=build-frontend /static/react/build /Anther/static
COPY --from=build-backend /Anther .
ENV DATABASE_URL=postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5433/${POSTGRES_DB}
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]