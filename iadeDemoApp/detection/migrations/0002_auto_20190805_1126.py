# Generated by Django 2.2.4 on 2019-08-05 11:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upload',
            name='image',
            field=models.ImageField(upload_to='upload/'),
        ),
    ]
