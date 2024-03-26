from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage

def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            return render(request, 'prediction_app/image_display.html', {
                'uploaded_file_url': uploaded_file_url
            })
    else:
        form = ImageUploadForm()
    return render(request, 'prediction_app/image_upload.html', {'form': form})
