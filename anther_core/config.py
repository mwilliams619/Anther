import os
from .cloud_settings import CLOUD_DATABASES, get_secret

# Enhanced environment detection
IN_PRODUCTION = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
DJANGO_ENV = os.environ.get("DJANGO_ENV", "development")
IN_CLOUDRUN = bool(os.getenv('K_SERVICE'))  
ON_CLOUDBUILD = bool(os.getenv('CLOUD_BUILD'))  

# Print for debugging
debug_msg = f"""
🔍 Current Environment:
   Production: {IN_PRODUCTION}
   Cloud Run: {IN_CLOUDRUN}
   Cloud Build: {ON_CLOUDBUILD}
   Django Environment: {DJANGO_ENV}
"""
print(debug_msg)

if IN_PRODUCTION or IN_CLOUDRUN or ON_CLOUDBUILD:
    DATABASES = CLOUD_DATABASES
    print(f"☁️ Using Cloud Database: {DATABASES['default']['NAME']} on {DATABASES['default']['HOST']}")
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'Anther_4',     
            'USER': 'mattwilliams',  
            'PASSWORD': '',         
            'HOST': 'localhost',    
            'PORT': '5433',         
        }
    }
    print(f"🖥️ Using Local Database: {DATABASES['default']['NAME']} on {DATABASES['default']['HOST']}")

# Make sure this is at the bottom:
print(f"Database Config: {DATABASES['default']['NAME']} on {DATABASES['default']['HOST']} (Engine: {DATABASES['default'].get('ENGINE')})")