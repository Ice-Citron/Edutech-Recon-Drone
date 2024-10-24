import React, { useState } from 'react';
import axios from 'axios';
import { db } from './firebase'; // Ensure correct import path
import { doc, getDoc } from 'firebase/firestore'; // Import Firestore functions

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');  // Trash count
  const [imageURL, setImageURL] = useState('');  // URL of uploaded image
  const [boundingBoxImageURL, setBoundingBoxImageURL] = useState('');  // URL of bounding box image
  const [metadata, setMetadata] = useState(null);  // State to hold metadata
  const [error, setError] = useState(''); // State to hold error messages

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPrediction('');  
    setImageURL('');  
    setBoundingBoxImageURL('');  
    setMetadata(null); 
    setError(''); 
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5050/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Extract the image URL, bounding box image URL, and trash count
      const { image_url, bounding_box_image_url, trash_count, document_id } = response.data;

      // Set the state with response data
      setImageURL(image_url);
      setBoundingBoxImageURL(bounding_box_image_url);  
      setPrediction(`Trash Count: ${trash_count}`);

      // Fetch metadata from Firestore
      const docRef = doc(db, 'trash_detection', document_id); // Use doc() instead of collection()
      const docSnap = await getDoc(docRef); // Use getDoc() to retrieve the document

      if (docSnap.exists()) {
        const docData = docSnap.data();
        setMetadata({
          date_created: docData.date_created,
          latitude: docData.latitude,
          longitude: docData.longitude,
        });
      } else {
        setError('No metadata found for the uploaded image.');
      }
      
    } catch (error) {
      console.error('Upload failed:', error);
      setError('Prediction failed. Please try again.');
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>
        Upload & Predict
      </button>

      {prediction && <p>{prediction}</p>}

      {imageURL && boundingBoxImageURL && (
        <div style={{ display: 'flex', marginTop: '20px' }}>
          <div style={{ marginRight: '10px' }}>
            <h3>Uploaded Image:</h3>
            <img src={imageURL} alt="Uploaded" style={{ width: '300px', height: 'auto' }} />
          </div>

          <div>
            <h3>Image with Bounding Boxes:</h3>
            <img src={boundingBoxImageURL} alt="Bounding Boxes" style={{ width: '300px', height: 'auto' }} />
          </div>
        </div>
      )}

      {metadata && (
        <div style={{ marginTop: '20px' }}>
          <h3>Extracted Metadata:</h3>
          <p>Date Created: {metadata.date_created}</p>
          <p>Latitude: {metadata.latitude}</p>
          <p>Longitude: {metadata.longitude}</p>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default FileUpload;
