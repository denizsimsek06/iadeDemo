from django.shortcuts import render, redirect
from detection.forms import UploadForm, PredictionForm
from detection.models import Upload, Prediction
from iadeDemoApp.predict import predict


def upload_image(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = UploadForm()
    return render(request, 'upload.html', {
        'form': form
    })


def prediction(request):
    predict()
    Prediction.objects.all().delete()
    img = 'prediction/prediction.jpg'
    pred = Prediction(image=img)
    form = PredictionForm(request.POST, instance=pred)
    form.save()
    return redirect('home')

def home(request):
    upload = Upload.objects.all()
    prediction = Prediction.objects.all()
    return render(request, 'home.html', {'upload': upload,
                                         'prediction': prediction})
