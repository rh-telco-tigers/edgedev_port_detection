# prediction_app/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from .image_utils import prepare_image, create_payload
import requests
from django.conf import settings
import logging
from PIL import Image as PilImage
import io

logger = logging.getLogger(__name__)

def image_upload_view(request):
    form = ImageUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        # Use the saved file path or work with the in-memory uploaded file
        image_path = fs.path(filename)

        try:
            # Prepare the image data
            image_data = prepare_image(image_path)
            payload = create_payload(image_data)

            # Send the request to the model server
            headers = {
                "Authorization": f"Bearer {settings.TOKEN}",
                "Content-Type": "application/json",
            }
            response = requests.post(settings.MODEL_SERVER_URL, json=payload, headers=headers, verify=settings.VERIFY_SSL)

            if response.status_code == 200:
                predictions = response.json()
                return render(request, 'prediction_app/image_display.html', {
                    'predictions': predictions,
                    'uploaded_file_url': uploaded_file_url,
                })
            else:
                logger.error("Failed to get predictions. Status code: %s, Response: %s", response.status_code, response.text)
                return render(request, 'prediction_app/error.html', {
                    'error': 'Failed to get predictions from the model server.',
                })
        except Exception as e:
            logger.error("Exception occurred: %s", str(e))
            return render(request, 'prediction_app/error.html', {
                'error': 'An exception occurred. Check server logs for more details.',
            })

    return render(request, 'prediction_app/image_upload.html', {'form': form})
