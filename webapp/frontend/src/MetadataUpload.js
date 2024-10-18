// MetadataUpload.js
import React, { useState } from 'react';
import axios from 'axios';

const MetadataUpload = () => {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMetadata(null); // Reset metadata when selecting a new file
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
      setMetadata(response.data); // Assuming the backend returns metadata in the response
      setError(''); // Reset error message
    } catch (error) {
      console.error('Upload failed:', error);
      setError('Metadata extraction failed. Please try again.');
      setMetadata(null); // Reset metadata on error
    }
  };

  return (
    <div>
      <h2>Upload Image for Metadata Extraction</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>
        Upload & Extract Metadata
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {metadata && (
        <div style={{ marginTop: '20px' }}>
          <h3>Extracted Metadata:</h3>
          <p>Latitude: {metadata.latitude}</p>
          <p>Longitude: {metadata.longitude}</p>
          <p>Date Created: {metadata.date_created}</p>
          <p>Image URL: <a href={metadata.image_url} target="_blank" rel="noopener noreferrer">{metadata.image_url}</a></p>
        </div>
      )}
    </div>
  );
};

export default MetadataUpload;
