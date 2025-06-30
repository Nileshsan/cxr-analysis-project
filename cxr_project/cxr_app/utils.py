from io import BytesIO
from reportlab.pdfgen import canvas

def generate_pdf_report(prediction):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)

    p.drawString(100, 750, "Chest X-Ray Report")
    p.drawString(100, 730, f"Prediction: {prediction}")

    # Add more details here as needed

    p.showPage()
    p.save()

    buffer.seek(0)
    return buffer

'''
import tensorflow as tf

def predict_disease(image_path):
    model = tf.keras.models.load_model('C:\\Users\\Nilesh\\mini_project_workspace\\nilesh.h5')
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    # Process the predictions and return the disease label
    predicted_label = decode_predictions(predictions)
    return predicted_label
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained model (update the path to your model)
model = load_model(r'C:\Users\Nilesh\mini_project_workspace\nilesh.h5')

# Define all_labels (update according to your model's output labels)
all_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
              'Consolidation', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'No Finding']

# Load and preprocess image function
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict on new data
img_path = r'C:\Users\Nilesh\mini_project_workspace\images_001\images\00000061_011.png'
new_image = load_and_preprocess_image(img_path)

# Make predictions
predictions = model.predict(new_image)

# Define different strategies for automated threshold
max_probability = np.max(predictions[0])  # Maximum probability from the predictions
mean_probability = np.mean(predictions[0])  # Mean of the probabilities
median_probability = np.median(predictions[0])  # Median of the probabilities

# Strategy 1: Set threshold to 50% of the highest probability
threshold_1 = 0.5 * max_probability

# Strategy 2: Set threshold as the mean or median probability (use one of them)
threshold_2 = mean_probability
# threshold_2 = median_probability  # Uncomment this line to use median instead of mean

# Map predictions to labels using the automated threshold
predicted_labels = [all_labels[i] for i in range(len(predictions[0])) if predictions[0][i] > threshold_2]

# Get the most probable disease
most_probable_index = np.argmax(predictions[0])
most_probable_disease = all_labels[most_probable_index]

# Print the results
print(f'Predicted labels (using threshold {threshold_2:.3f}): {predicted_labels}')
print(f'Most probable disease: {most_probable_disease}')


plt.imshow(new_image[0])  # Display the image
plt.title(f'Predicted labels: {predicted_labels}')
plt.axis('off')
plt.show()
