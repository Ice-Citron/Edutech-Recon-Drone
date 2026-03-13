# Recon Drone

**Recon Drone** is a full-stack project for detecting and mapping environmental waste using drone imagery and computer vision.

The system processes drone images, detects trash using a YOLO-based object detection model, and visualizes waste density on an interactive map.

This project was built for the **Edutech Asia 2024 Planet Protectors competition**, where it won **first place**.

---

## Features

- **Trash Detection:** Uses a YOLOv11 object detection model to detect and count waste objects in drone imagery.

- **Geolocation:** Extracts EXIF metadata (GPS coordinates and timestamps) from uploaded images to determine where the image was captured.

- **Heatmap Visualization:** Displays waste density on an interactive map using the Google Maps Visualization API.

- **Cloud Storage:** Images and detection results are stored in Firebase. Firestore stores metadata and detection results, while Cloud Storage stores uploaded images and processed outputs.
---

## Tech Stack

| Category     | Technology                                     |
| :----------- | :--------------------------------------------- |
| **Frontend** | React.js, Vite, Google Maps API, Axios         |
| **Backend**  | Python, Flask, Gunicorn, PIL (EXIF)            |
| **AI/ML**    | YOLOv11 (Ultralytics)                          |
| **Cloud/DB** | Firebase (Firestore, Cloud Storage, Admin SDK) |

---

## System Architecture

1.  User uploads a drone image via the React frontend.
2.  The Flask backend extracts GPS metadata and runs the YOLOv11 inference engine.
3.  Original images and processed versions (with bounding boxes) are stored in Firebase Cloud Storage; metadata is saved to Firestore.
4.  The frontend fetches historical data from Firestore and renders markers and heatmaps on a customized Google Map.

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js & npm
- Google Maps API Key
- Firebase Service Account Key

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your `firebase-adminsdk.json` to the `backend` folder.
4. Run the server:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5050 app:app
   ```

### Frontend Setup

1. Navigate to the `DroneDataVisualization` directory:
   ```bash
   cd DroneDataVisualization
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file and add your Google Maps API key:
   ```env
   VITE_GOOGLE_MAPS_API_KEY=your_api_key_here
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

---

## Preview

<img width="1512" alt="EcoDrone Dashboard" src="https://github.com/user-attachments/assets/1b98faa7-382c-4f82-a061-e870f752fb0e" />

---

## Future Improvements

- [ ] Direct integration with DJI/Mavic SDKs for live mapping.
- [ ] Tracking waste movement over time using historical data.
- [ ] Optimizing the YOLO model for on-board drone processing (NVIDIA Jetson).
