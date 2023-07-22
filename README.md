Setup local db environment
Make sure you have postgres and Postico2 setup in your environment
Make a postgres server
1. Copy and past localsettingstemplate.py content into localsettings.py
2. Connect localsettings.py with your local postgres environment
3. In launch.json update to your local manage.py path
4. Create local venv
5. download required packages pip install -r requirements.txt
6. populate db with current models
    6.1 run python manage.py makemigrations
    6.2 run python manage.py migrate
