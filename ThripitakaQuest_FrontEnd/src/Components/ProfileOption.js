// ProfileOptionsPopup.js
import React from 'react';
import { Modal, Button } from 'react-bootstrap';
import "./ProfileOption.css";

export default function ProfileOptionsPopup({ show, onHide, onSettingsClick, onLogoutClick }) {
  
  return (
    <Modal show={show} onHide={onHide} className="model">
      <Modal.Body>
        <div className="d-flex flex-column">
          <button  className="settingsBtn" onClick={onSettingsClick}>
            Change Settings
          </button>

          <Button className="logOutBtn" variant="danger" onClick={onLogoutClick}>
            Logout
          </Button>
        </div>
      </Modal.Body>
    </Modal>
  );
}
