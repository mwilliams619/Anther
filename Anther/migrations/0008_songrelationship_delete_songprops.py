# Generated by Django 4.2 on 2023-07-25 02:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("Anther", "0007_playlist_uri"),
    ]

    operations = [
        migrations.CreateModel(
            name="SongRelationship",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("similarity_score", models.FloatField()),
                (
                    "song_a",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="related_songs_a",
                        to="Anther.song",
                    ),
                ),
                (
                    "song_b",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="related_songs_b",
                        to="Anther.song",
                    ),
                ),
            ],
        ),
        migrations.DeleteModel(name="SongProps",),
    ]
