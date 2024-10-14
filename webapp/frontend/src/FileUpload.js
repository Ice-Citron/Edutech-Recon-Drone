import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');

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

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>
        Upload & Predict
      </button>
      {prediction && <p>{prediction}</p>}
    </div>
  );
};

export default FileUpload;
