from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS  
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import io
import firebase_admin
from firebase_admin import credentials, firestore, storage
import base64
import matplotlib.pyplot as plt
import numpy as np

# Initialize Flask app and Firebase
app = Flask(__name__)
CORS(app)  

cred = credentials.Certificate('airecondrone-firebase-adminsdk-1oiv0-cfdb4ae33a.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'airecondrone.appspot.com'
})

db = firestore.client()
bucket = storage.bucket()

# Load YOLO model for trash detection
print("Model Loaded")
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

def get_exif_data(image):
    """Extract EXIF data from an image."""
    exif_data = {}
    try:
        exif = image._getexif()  
        if exif:
            for tag, value in exif.items():
                decoded_tag = TAGS.get(tag, tag)
                exif_data[decoded_tag] = value
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
    return exif_data

def get_gps_data(exif_data):
    """Extract GPS data from EXIF."""
    gps_data = {}
    if "GPSInfo" in exif_data:
        for key in exif_data["GPSInfo"].keys():
            decoded_key = GPSTAGS.get(key, key)
            gps_data[decoded_key] = exif_data["GPSInfo"][key]
    return gps_data

def convert_to_degrees(value):
    """Convert GPS coordinates to degrees."""
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def extract_metadata(image):
    """Extract GPS and date metadata from the image."""
    exif_data = get_exif_data(image)
    gps_data = get_gps_data(exif_data)

    latitude = longitude = None
    if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
        latitude = convert_to_degrees(gps_data["GPSLatitude"])
        if gps_data["GPSLatitudeRef"] != "N":
            latitude = -latitude

    if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
        longitude = convert_to_degrees(gps_data["GPSLongitude"])
        if gps_data["GPSLongitudeRef"] != "E":
            longitude = -longitude

    date_created = exif_data.get("DateTimeOriginal", "Unknown")

    return latitude, longitude, date_created

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        file_content = file.read()
        if not file_content:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        # Upload the image to Firebase Storage
        blob = bucket.blob(file.filename)
        blob.upload_from_file(io.BytesIO(file_content), content_type=file.content_type)

        # Construct public URL for the uploaded image
        image_path = file.filename.replace(" ", "%20")
        image_url = f"https://firebasestorage.googleapis.com/v0/b/airecondrone.appspot.com/o/{image_path}?alt=media"

        # Open and verify the image format
        img = Image.open(io.BytesIO(file_content))
        img.verify()  # Validate the image

        # Reopen the image after verification
        img = Image.open(io.BytesIO(file_content))

        # Run YOLO inference to detect trash and get the result image with bounding boxes
        results = model.predict(img, save=False, verbose=False)

        # Convert the result to an image with bounding boxes (as a NumPy array)
        result_image = results[0].plot()  # This returns a NumPy array

        # Convert NumPy array (result_image) to a PIL image
        result_image_pil = Image.fromarray(np.uint8(result_image))

        # Save the result image to a BytesIO object
        img_with_boxes_io = io.BytesIO()
        result_image_pil.save(img_with_boxes_io, format='JPEG')
        img_with_boxes_io.seek(0)

        # Optionally, upload the modified image back to Firebase (or return as base64)
        box_blob = bucket.blob(f"bounding_box_{file.filename}")
        box_blob.upload_from_file(img_with_boxes_io, content_type="image/jpeg")

        # Construct URL for the modified image with bounding boxes
        box_image_path = f"bounding_box_{file.filename}".replace(" ", "%20")
        box_image_url = f"https://firebasestorage.googleapis.com/v0/b/airecondrone.appspot.com/o/{box_image_path}?alt=media"

        # Count the number of bounding boxes detected
        bounding_boxes_count = len(results[0].boxes) if results and results[0].boxes else 0

        latitude, longitude, date_created = extract_metadata(img)

        doc_ref = db.collection('trash_detection').add({
            'image_url': image_url,
            'bounding_box_image_url': box_image_url,
            'trash_count': bounding_boxes_count,
            'latitude': latitude,
            'longitude': longitude,
            'date_created': date_created
        })

        _, doc_ref = doc_ref
        document_id = doc_ref.id  

        # Return JSON response with both the original image URL and the bounding box image URL
        return jsonify({
            'image_url': image_url,
            'bounding_box_image_url': box_image_url,
            'trash_count': bounding_boxes_count,
            'document_id': document_id
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)