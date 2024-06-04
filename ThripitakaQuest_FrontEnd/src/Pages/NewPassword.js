import React, { useState } from 'react';
import { getAuth, confirmPasswordReset } from 'firebase/auth';
import { useLocation, useNavigate } from 'react-router-dom';
import { Form, Button, Alert } from 'react-bootstrap';
import './NewPassword.css';

export default function NewPassword() {
  const [newPassword, setNewPassword] = useState('');
  const [rePassword, setRePassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showRePassword, setShowRePassword] = useState(false);
  const [alert, setAlert] = useState(null);

  const auth = getAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const showAlert = (message, variant = 'danger') => {
    setAlert({ message, variant });
    setTimeout(() => setAlert(null), 2000);
  };

  const validatePassword = (password) => {
    // Password complexity validation logic
    const passwordPolicy = "Password must include at least one lower letter, upper letter, number & non-alphanumeric character (!@#$%&*).";

    if (password.length < 8 || password.length > 14) {
      showAlert(`Password must be between 8 and 14 characters long. ${passwordPolicy}`);
      return false;
    }

    if (!/[a-z]/.test(password)) {
      showAlert(passwordPolicy);
      return false;
    }

    if (!/[A-Z]/.test(password)) {
      showAlert(passwordPolicy);
      return false;
    }

    if (!/[0-9]/.test(password)) {
      showAlert(passwordPolicy);
      return false;
    }

    if (!/[^\w]/.test(password)) {
      showAlert(passwordPolicy);
      return false;
    }

    return true;
  };

  // Enable to user to check password by showing
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const toggleRePasswordVisibility = () => {
    setShowRePassword(!showRePassword);
  };

  const handlePasswordReset = () => {
    // Check the validity of the password
    const passwordIsValid = validatePassword(newPassword);
    if (!passwordIsValid) {
      return;
    }

    // Check whether user password and confirm password are equals
    if (newPassword !== rePassword) {
      showAlert('Passwords do not match.');
      return;
    }

    // Extract oobcode status to validate the email link
    const searchParams = new URLSearchParams(location.search);
    const oobCode = searchParams.get('oobCode');

    // If oobcode is not a valid one display an alert
    if (!oobCode) {
      showAlert('Invalid reset link. Please request a new one.');
      return;
    }

    // Reset the old password and display a message
    confirmPasswordReset(auth, oobCode, newPassword)
      .then(() => {
        showAlert('Password reset successful. You can now log in with your new password.', 'success');
        navigate('/', { replace: true })
      })
      .catch((error) => {
        // If link is used or timed out, display a message
        if (error.code === 'auth/invalid-action-code') {
          showAlert('Link to reset your password has expired or has already been used.');
        } else {
          showAlert(`Error resetting password: ${error.message}`);
        }
      });
  };

  return (
    <div className='ContainerNewPassword'>
      <div className='newResetPassword'>
        <h2>New Password</h2>
        <Form onSubmit={(e) => e.preventDefault()}>
          <Form.Group controlId="password" className="newPasswordGroup">
            <div className="password-input-wrapper">
              <Form.Control
                type={showPassword ? 'text' : 'password'}
                size="lg"
                placeholder="Password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />
              <span className="toggle-password-button" onClick={togglePasswordVisibility}>
                {showPassword ? 'Hide' : 'Show'}
              </span>
            </div>
          </Form.Group>

          <Form.Group controlId="confirmPassword" className="newPasswordGroup">
            <div className="password-input-wrapper">
              <Form.Control
                size="lg"
                type={showRePassword ? 'text' : 'password'}
                placeholder="Confirm Password"
                value={rePassword}
                onChange={(e) => setRePassword(e.target.value)}
              />
              <span className="toggle-password-button" onClick={toggleRePasswordVisibility}>
                {showRePassword ? 'Hide' : 'Show'}
              </span>
            </div>
          </Form.Group>

          <Button variant="light" className="resetPasswordButton" onClick={handlePasswordReset}>
            Reset Password
          </Button>

          {alert && (
            <Alert variant={alert.variant} className="mt-3 errorAlert">
              {alert.message}
            </Alert>
          )}
        </Form>
      </div>
    </div>
  );
}
