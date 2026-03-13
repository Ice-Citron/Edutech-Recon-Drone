# EcoDrone: AI-Powered Environmental Surveillance 🛰️🚮

[![Edutech Asia 2024 Champion](https://img.shields.io/badge/Award-Edutech%20Asia%202024%20Champion-gold?style=for-the-badge)](https://www.edutech.asia/)
[![Python](https://img.shields.io/badge/Backend-Python%20%7C%20Flask-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
[![YOLOv11](https://img.shields.io/badge/AI-YOLOv11-green?style=for-the-badge)](https://github.com/ultralytics/ultralytics)

**EcoDrone** is an award-winning full-stack solution designed to automate environmental monitoring using drone imagery and computer vision. Developed for the **Edutech Asia 2024 Planet Protectors** competition, this prototype leverages state-of-the-art AI to detect, quantify, and map environmental waste in real-time.

---

## 🌟 Key Achievements

- **Champion**: Edutech Asia 2024 Planet Protectors.
- **Keynote**: Presented to an audience of 1,000+ industry leaders and educators.

---

## 🚀 Core Features

### 🔍 Automated Trash Detection (YOLOv11)

Utilizes the latest **YOLOv11 CNN model** to identify and count waste particles in high-resolution drone imagery with high precision and low latency.

### 📍 Geospatial Mapping & Metadata Extraction

Automatically extracts **EXIF metadata** (GPS coordinates and timestamps) from uploaded images to precisely locate waste on an interactive map.

### 📊 Trash Density Heatmaps

Generates dynamic heatmaps using the **Google Maps API Visualization Library**, allowing authorities to identify "hot zones" and optimize cleanup resources.

### ☁️ Scalable Cloud Architecture

Integrated with **Firebase** (Firestore and Cloud Storage) for real-time data persistence, image hosting, and seamless cross-platform synchronization.

---

## 🛠️ Tech Stack

| Category     | Technology                                     |
| :----------- | :--------------------------------------------- |
| **Frontend** | React.js, Vite, Google Maps API, Axios         |
| **Backend**  | Python, Flask, Gunicorn, PIL (EXIF)            |
| **AI/ML**    | YOLOv11 (Ultralytics)                          |
| **Cloud/DB** | Firebase (Firestore, Cloud Storage, Admin SDK) |

---

## 🏗️ System Architecture

1.  **Ingestion**: User uploads a drone image via the React frontend.
2.  **Processing**: The Flask backend extracts GPS metadata and runs the YOLOv11 inference engine.
3.  **Storage**: Original images and processed versions (with bounding boxes) are stored in Firebase Cloud Storage; metadata is saved to Firestore.
4.  **Visualization**: The frontend fetches historical data from Firestore and renders markers and heatmaps on a customized Google Map.

---

## ⚙️ Installation & Setup

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
   _(Note: Ensure `ultralytics`, `flask`, `flask-cors`, `firebase-admin`, and `Pillow` are installed)_
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

## 📸 Preview

<img width="1512" alt="EcoDrone Dashboard" src="https://github.com/user-attachments/assets/1b98faa7-382c-4f82-a061-e870f752fb0e" />

---

## 🔮 Future Roadmap

- [ ] **Real-time Telemetry**: Direct integration with DJI/Mavic SDKs for live mapping.
- [ ] **Temporal Analysis**: Tracking waste movement over time using historical data.
- [ ] **Edge Deployment**: Optimizing the YOLO model for on-board drone processing (NVIDIA Jetson).

---

_Developed with ❤️ for the Planet Protectors Initiative._
