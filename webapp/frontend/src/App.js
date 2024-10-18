import React from 'react';
import FileUpload from './FileUpload'; 
import MetadataUpload from './MetadataUpload';

const App = () => {
  return (
    <div>
      <h1>Image Processing App</h1>
      <FileUpload /> // Prediction Page
      <MetadataUpload /> // Metadata Extraction Page
    </div>
  );
};

export default App;