# Generated by Django 2.2.4 on 2019-08-06 06:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0003_auto_20190805_1134'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='image',
            field=models.ImageField(upload_to='prediction/'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='image',
            field=models.ImageField(upload_to='upload/'),
        ),
    ]
