# Generated by Django 5.0.4 on 2024-10-17 08:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_alter_uploadedimage_detected_objects'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='processing_time',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
