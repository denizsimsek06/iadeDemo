from django.core.files.storage import default_storage
from django.db import models

# Create your models here.



def upload_location(instance, filename):
    filebase, extension = filename.split('.')
    extension = 'jpg'
    instance.name = 'upload'
    Upload.objects.all().delete()
    if default_storage.exists('upload/%s.%s' % (instance.name, extension)):
        default_storage.delete('upload/%s.%s' % (instance.name, extension))
    return 'upload/%s.%s' % (instance.name, extension)


def prediction_location(instance, filename):
    filebase, extension = filename.split('.')
    instance.name = 'prediction'
    extension = 'jpg'
    Prediction.objects.all().delete()
    if default_storage.exists('prediction/%s.%s' % (instance.name, extension)):
        default_storage.delete('prediction/%s.%s' % (instance.name, extension))
    return 'prediction/%s.%s' % (instance.name, extension)


class Upload(models.Model):
    image = models.ImageField(upload_to=upload_location)


class Prediction(models.Model):
    image = models.ImageField(upload_to=prediction_location)