import os
import django
from django.conf import settings
from django.db import connection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anther_core.localsettings')  # Replace with your actual settings module
django.setup()

print(f"👉 Production Mode: {settings.IN_PRODUCTION}")
print(f"👉 Database: {settings.DATABASES['default']['NAME']}")
print(f"👉 User: {settings.DATABASES['default']['USER']}")
print(f"👉 Host: {settings.DATABASES['default']['HOST']}")

try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        print("✅ Successfully connected to Django configured database!")
except Exception as e:
    print(f"❌ Django DB Connection Error: {e}")