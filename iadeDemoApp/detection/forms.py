from django import forms
from .models import Upload, Prediction


class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ['image']


class PredictionForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ['image']
