# Generated by Django 5.0.4 on 2024-10-03 06:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_alter_uploadedimage_detected_objects'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedimage',
            name='detected_objects',
            field=models.TextField(blank=True, null=True),
        ),
    ]
