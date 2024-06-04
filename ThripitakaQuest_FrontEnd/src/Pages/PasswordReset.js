import React, { useState, useEffect } from 'react';
import { getAuth, sendPasswordResetEmail } from 'firebase/auth';
import './PasswordReset.css';
import { Alert } from 'react-bootstrap';

export default function ResetPassword() {
  const [email, setEmail] = useState('');
  const [resetStatus, setResetStatus] = useState(null);

  const handleResetPassword = () => {
    const auth = getAuth();
    // Send an email to user give email
    sendPasswordResetEmail(auth, email)
      .then(() => {
        setResetStatus({
          success: true,
          message: 'Password reset email sent successfully. Check your email.',
        });
        setEmail(''); 
      })
      // Display if any error occured
      .catch((error) => {
        if (error.code === 'auth/user-not-found') {
          setResetStatus({
            success: false,
            message: 'User with this email does not exist. Please sign up.',
          });
        } else if (error.code === 'auth/invalid-email') {
          setResetStatus({
            success: false,
            message: 'Please enter a valid email.',
          });
        } else if (error.code === 'auth/missing-email') {
          setResetStatus({
            success: false,
            message: 'Please enter an email to send reset email.',
          });
        } else if (error.code === 'auth/network-request-failed') {
          setResetStatus({
            success: false,
            message: 'Check internet connection and try again.',
          });
        } else {
          setResetStatus({
            success: false,
            message: 'Error sending password reset email. Please try again.',
          });
        }
        setEmail(''); 
      });
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setResetStatus(null);
    }, 2000);

    return () => {
      clearTimeout(timer);
    };
  }, [resetStatus]);

  return (
    <div className="containerPasswordResetRequest">
      <div className="passwordResetRequest">
        <h2>Reset Password</h2>
        <form className="emailSend">
          <label>
            <input
              className="email inputs"
              type="email"
              value={email}
              placeholder="Email"
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <button type="button" className="resetPasswordBtn" onClick={handleResetPassword}>
            Reset Password
          </button>
        </form>
        {resetStatus && (
          <Alert variant={resetStatus.success ? 'success' : 'danger'}>
            {resetStatus.message}
          </Alert>
        )}
      </div>
    </div>
  );
}
