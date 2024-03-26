# prediction_app/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from .image_utils import prepare_image, create_payload
from .utils import get_categories_from_json, select_top_predictions_per_group
import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def image_upload_view(request):
    form = ImageUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        try:
            image_data = prepare_image(fs.path(filename))  # Preprocess the image
            payload = create_payload(image_data)  # Prepare the payload for the model server
            
            headers = {
                "Authorization": f"Bearer {settings.TOKEN}",
                "Content-Type": "application/json",
            }
            
            response = requests.post(settings.MODEL_SERVER_URL, json=payload, headers=headers, verify=settings.VERIFY_SSL)
            
            if response.status_code == 200:
                predictions_response = response.json()
                
                # Assuming 'predictions_response' structure is {'outputs': [{'data': raw_predictions}]}
                raw_predictions = predictions_response['outputs'][0]['data']
                categories = get_categories_from_json('../../dataset/custom-data/result.json')
                
                top_predictions = select_top_predictions_per_group(raw_predictions, categories, n=8)
                
                return render(request, 'prediction_app/image_display.html', {
                    'predictions': top_predictions,
                    'uploaded_file_url': uploaded_file_url,
                })
            else:
                logger.error("Failed to get predictions. Status code: %s, Response: %s", response.status_code, response.text)
                return render(request, 'prediction_app/error.html', {
                    'error': 'Failed to get predictions from the model server.',
                })
        except Exception as e:
            logger.error("Exception occurred: %s", str(e), exc_info=True)
            return render(request, 'prediction_app/error.html', {
                'error': 'An exception occurred. Check server logs for more details.',
            })

    return render(request, 'prediction_app/image_upload.html', {'form': form})
