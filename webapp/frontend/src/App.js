import React, { useState } from 'react';
import './App.css';

function App() {
  const [fileContent, setFileContent] = useState('');

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'text/plain') {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target.result);
      };
      reader.readAsText(files[0]);
    } else {
      alert('Please drop a valid .txt file.');
    }
  };

  return (
    <div className="App">
      <div
        id="drop-area"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="drop-area"
      >
        <p>Drag and drop a .txt file here</p>
      </div>
      <div id="file-content" className="file-content">
        {fileContent}
      </div>
    </div>
  );
}

export default App;
