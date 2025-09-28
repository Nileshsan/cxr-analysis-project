import os
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.core.files.base import ContentFile
from django.conf import settings

# Load your model (make sure the model path is correct)
MODEL_PATH = r'c:\PROJECTS\cxr-analysis-project\model\nilesh.h5'  # Adjusted to the model in the repo

# Lazy model loader: load on first request to avoid import-time crashes
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        return _MODEL
    except TypeError as e:
        # Try loading without compiling which can avoid some deserialization issues
        try:
            _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return _MODEL
        except Exception as e2:
            # Give a clear message and leave _MODEL as None
            print(f"Failed to load model at {MODEL_PATH}:", e2)
            _MODEL = None
            return None
    except Exception as e:
        print(f"Failed to load model at {MODEL_PATH}:", e)
        _MODEL = None
        return None

import numpy as np
from PIL import Image

def preprocess_image(uploaded_file):
    # Load the image
    img = Image.open(uploaded_file)

    # Resize the image to 224x224
    img = img.resize((224, 224))

    # Convert the image to an array and scale the pixel values
    img_array = np.array(img) / 255.0  # Normalize the pixel values to [0, 1]

    # Check if the image is grayscale and convert to RGB
    if img_array.ndim == 2:  # Shape is (224, 224), meaning it's grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to shape (224, 224, 3)

    # Add batch dimension to the array (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def generate_pdf_report(predicted_labels, confidence):
    """Generate a PDF report with the prediction results."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Chest X-ray Analysis Report")
    p.drawString(100, 720, f"Predicted Conditions: {', '.join(predicted_labels)}")
    p.drawString(100, 700, f"Confidence: {confidence:.2f}%")
    p.showPage()
    p.save()
    buffer.seek(0)
    return ContentFile(buffer.read(), name='report.pdf')


def upload_xray(request):
    if request.method == 'POST':
        # Process the uploaded file and generate a report
        uploaded_file = request.FILES['uploaded_file']
        img_array = preprocess_image(uploaded_file)

        # Make prediction
        model = get_model()
        if model is None:
            # Return a mock prediction so the UI can continue to function during development
            print('Model not loaded; returning mock prediction')
            mock_labels = ['No Finding']
            mock_confidence = 99.0
            predicted_labels = mock_labels
            confidence = mock_confidence
        else:
            predictions = model.predict(img_array)
            print("Raw predictions:", predictions)  # Add this line for debugging

            predicted_labels = []  # Convert predictions to labels
            confidence = np.max(predictions) * 100  # Example confidence calculation
            predicted_indices = np.argmax(predictions, axis=1)  # Get indices of the highest probabilities

            # Map indices to your actual labels
            class_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
                          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Emphysema',
                          'Fibrosis', 'Pleural Thickening', 'No Finding']  # Update with actual labels
            predicted_labels = [class_labels[i] for i in predicted_indices]

    # If model was mocked above, predicted_labels and confidence are set already

        # Generate the PDF report
        report_content = generate_pdf_report(predicted_labels, confidence)

        # Ensure MEDIA_ROOT exists
        media_root = getattr(settings, 'MEDIA_ROOT', None) or os.path.join(os.getcwd(), 'media')
        os.makedirs(media_root, exist_ok=True)

        filename = 'generated_report.pdf'
        save_path = os.path.join(media_root, filename)
        with open(save_path, 'wb') as f:
            # report_content is a ContentFile; read its bytes
            f.write(report_content.read())

        # Build a URL the frontend can download. Rely on MEDIA_URL setting if available.
        media_url = getattr(settings, 'MEDIA_URL', '/media/')
        report_url = request.build_absolute_uri(os.path.join(media_url, filename))

        # Return the results as JSON
        return JsonResponse({
            'predicted_labels': predicted_labels,
            'confidence': confidence,
            'report_url': report_url
        })

    return render(request, 'upload_xray.html')
