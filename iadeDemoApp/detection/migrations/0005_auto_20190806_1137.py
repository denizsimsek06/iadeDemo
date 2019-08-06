# Generated by Django 2.2.4 on 2019-08-06 11:37

import detection.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0004_auto_20190806_0622'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='count',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='image',
            field=models.ImageField(upload_to=detection.models.prediction_location),
        ),
        migrations.AlterField(
            model_name='upload',
            name='image',
            field=models.ImageField(upload_to=detection.models.upload_location),
        ),
    ]
