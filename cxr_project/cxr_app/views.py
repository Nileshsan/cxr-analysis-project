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

# Load your model (make sure the model path is correct)
MODEL_PATH = r'C:\Users\Nilesh\mini_project_workspace\nilesh.h5'  # Adjust the path to your model
model = tf.keras.models.load_model(MODEL_PATH)

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

        # Generate the PDF report
        report_content = generate_pdf_report( predicted_labels, confidence)
        report_url = r'C:\Users\Nilesh\mini_project_workspace\store_pdf'  # Save the report or serve it dynamically

        # Save report to the media directory or serve it
        with open(os.path.join('media', 'generated_report.pdf'), 'wb') as f:
            f.write(report_content.read())

        # Return the results as JSON
        return JsonResponse({
            'predicted_labels': predicted_labels,
            'confidence': confidence,
            'report_url': report_url
        })

    return render(request, 'upload_xray.html')
