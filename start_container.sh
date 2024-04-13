#!/bin/bash

# Prompt the user for database name, username, and password
echo "Enter the database name:"
read POSTGRES_DB
echo "Enter the Postgres username:"
read POSTGRES_USER
echo "Enter the Postgres password:"
read -s POSTGRES_PASSWORD
echo

# Export the environment variables
export POSTGRES_DB
export POSTGRES_USER
export POSTGRES_PASSWORD

# Build the Docker images
docker-compose build 

# Start the Docker containers
docker-compose up -d