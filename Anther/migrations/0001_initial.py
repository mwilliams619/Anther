# Generated by Django 4.2 on 2023-07-23 13:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="SongProps",
            fields=[
                ("id", models.BigAutoField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=255, unique=True)),
                ("danceability", models.IntegerField()),
                ("energy", models.IntegerField()),
                ("mode", models.IntegerField()),
                ("valence", models.IntegerField()),
                ("tempo", models.IntegerField()),
                ("uri", models.TextField()),
                ("key", models.IntegerField()),
                ("popularity", models.IntegerField()),
                ("genre", models.CharField(max_length=255)),
            ],
        ),
    ]
