Setup local db environment
Make sure you have postgres and Postico2 setup in your environment
Make a postgres server
1. go to settings.py
2. Change lines 84 to 92 to connect with your local postgres environment
3. Create local venv
4. populate db with current models
    4.1 run python manage.py makemigrations
    4.2 run python manage.py migrate
