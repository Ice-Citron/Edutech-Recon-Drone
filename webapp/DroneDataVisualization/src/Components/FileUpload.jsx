// Update the import statement at the top of the file
import React, { useState } from 'react';
import axios from 'axios';
import { db } from './firebase'; // Ensure correct import path
import { doc, getDoc } from 'firebase/firestore'; // Import Firestore functions

// Destructure onUploadSuccess from props
const FileUpload = ({ onUploadSuccess }) => {  // Fix here
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState(''); // State to hold the file name
  const [prediction, setPrediction] = useState('');  // Trash count
  const [imageURL, setImageURL] = useState('');  // URL of uploaded image
  const [boundingBoxImageURL, setBoundingBoxImageURL] = useState('');  // URL of bounding box image
  const [metadata, setMetadata] = useState(null);  // State to hold metadata
  const [error, setError] = useState(''); // State to hold error messages

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setFileName(selectedFile ? selectedFile.name : ''); // Set file name
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

        onUploadSuccess();  // This will now work correctly
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
      {/* Hide the default file input */}
      <input 
        type="file" 
        onChange={handleFileChange} 
        style={{ display: 'none' }} // Hide the input
        id="file-upload" // Add an ID for the custom button to reference
      />
      {/* Create a custom button to trigger the file input */}
      <label htmlFor="file-upload" style={{
        display: 'flex',                 // Use flexbox
        flexDirection: 'column',         // Arrange children in a column
        justifyContent: 'center',        // Center content vertically
        alignItems: 'center',            // Center content horizontally
        height: '20vh',
        width: '25vw',
        color: 'black',                  // Set text color
        border: 'black dotted',
        background: 'lightgray',
        textAlign:'center',
        cursor: 'pointer',
        marginBottom: '10px',           // Space between file input and upload button
        position: 'absolute',
        margin: '20px 0 0 0',
        right: '50%',
        transform: 'translateX(50%)'
      }}>
          Choose File
          {fileName && <p style={{ margin: '10px 0 0 0', fontSize:'10pt' }}>Selected File: {fileName}</p>} {/* Display selected file name */}
      </label>

      <button style={{ position: 'absolute', 
                    background: '#007bff', color: 'white', 
                    right: '50%', bottom:'20px',
                    transform: 'translateX(50%)'}}
                    onClick={handleUpload}>
        Upload
      </button>
    </div>
  );
};

export default FileUpload;
