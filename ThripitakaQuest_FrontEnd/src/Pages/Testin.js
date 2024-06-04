import React from 'react';
import { useLocation } from 'react-router-dom';

function Testin() {
  const location = useLocation();
  const photoURL = location.state?.photoURL || null;
  const email = location.state?.email || null;
  const userID = location.state?.userID || null;

  return (
    <div>
      <h1>Welcome to the Chat</h1>
      {photoURL && email && (
        <div className="user-profile">
          <img src={photoURL} alt="User Profile" />
          <div className="user-details">
            <h2>{userID}</h2>
            <p>{email}</p>
          </div>
        </div>
      )}

      {/* Your chat content goes here */}
    </div>
  );
}

export default Testin;
