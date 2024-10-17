import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [recentImage, setRecentImage] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPrediction('');  // Reset prediction when selecting a new file
  };

  const handleUpload = async () => {
    if (!file) {
      setPrediction('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5050/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(`Predicted Class: ${response.data.predicted_class}`);
    } catch (error) {
      console.error('Upload failed:', error);
      setPrediction('Prediction failed. Please try again.');
    }
  };

  const fetchRecentImage = async () => {
    try {
      const response = await axios.get('http://localhost:5050/recent-image');
      setRecentImage(response.data.image_url);
    } catch (error) {
      console.error('Failed to fetch recent image:', error);
      setRecentImage('');
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>
        Upload & Predict
      </button>
      <button onClick={fetchRecentImage} style={{ marginLeft: '10px' }}>
        Show Recent Image
      </button>
      {prediction && <p>{prediction}</p>}
      {recentImage && (
        <div style={{ marginTop: '20px' }}>
          <h3>Most Recent Image:</h3>
          <img src={recentImage} alt="Recent" style={{ width: '300px', height: 'auto' }} />
        </div>
      )}
    </div>
  );
};

export default FileUpload;
