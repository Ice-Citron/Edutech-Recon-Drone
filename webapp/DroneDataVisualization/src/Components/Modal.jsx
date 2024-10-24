// Modal.jsx
import React from 'react';
import './Modal.css'; // Ensure you have styles for your modal

const Modal = ({ isOpen, onClose, children }) => {
    if (!isOpen) return null; // Don't render the modal if not open

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <button style={{ position: 'absolute', 
                    background: 'white', color: 'black', 
                    right: '0', top:'0'}} onClick={onClose}>X</button>
                {children}
            </div>
        </div>
    );
};

export default Modal;
