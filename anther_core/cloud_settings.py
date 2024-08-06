import os
from google.auth.exceptions import DefaultCredentialsError
os.environ['GOOGLE_CLOUD_PROJECT'] = "studious-loader-429917-e3"
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', "studious-loader-429917-e3")

def get_secret(secret_id, default_value=None, version_id="latest"):
    try:
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(name=name)
        return response.payload.data.decode('UTF-8').strip()
    except DefaultCredentialsError:
        print(f"⚠️ No Google credentials. Using default for {secret_id}: {default_value}")
        return default_value
    except Exception as e:
        print(f"⚠️ Error retrieving {secret_id}: {e}")
        return default_value

INSTANCE_CONNECTION_NAME = "studious-loader-429917-e3:us-central1:anther-1"

print(f"🔄 Loading Cloud Settings (Project ID: {project_id})")
print(f"☁️ Cloud SQL Instance: {INSTANCE_CONNECTION_NAME}")

CLOUD_DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': get_secret("DB_NAME", "Anther-1"),
        'USER': get_secret("DB_USER", "anther_admin"),
        'PASSWORD': get_secret("DB_PASSWORD", ""),
        'HOST': f'/cloudsql/{INSTANCE_CONNECTION_NAME}',
        'PORT': '5432',
    }
}

print(f"☁️ Cloud Database Config: {CLOUD_DATABASES['default']['NAME']} on {CLOUD_DATABASES['default']['HOST']}")