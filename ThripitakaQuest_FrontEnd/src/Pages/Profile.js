import React, { useState, useEffect } from 'react';
import { Col, Button, Row, Form, Modal } from 'react-bootstrap';
import { useAuthState } from 'react-firebase-hooks/auth';
import { updateProfile as updateProfileFirebase, updatePassword, reauthenticateWithPopup,
  GoogleAuthProvider,getAuth, reauthenticateWithCredential, EmailAuthProvider,} from 'firebase/auth';
import { doc, getDoc, getDocs, updateDoc, deleteDoc, collection, query, where, deleteObject } from 'firebase/firestore';
import { list, ref, ref as storageRef, uploadBytes, getDownloadURL, deleteObject as deleteStorageObject } from 'firebase/storage';
import { auth, db, storage } from './firebase';
import { useLocation, useNavigate } from 'react-router-dom';
import { useUserContext } from '../UserContext';
import userIcon from "../assets/userIcon.png";
import './Profile.css';

export default function UserProfile() {
  const location = useLocation();
  const { users,updateUser } = useUserContext();
  const { email, firstName, lastName, imageUrl } = users;
  // Define the state variables
  const [user, loading, error] = useAuthState(auth);
  const [userProfile, setUserProfile] = useState({
    firstName: firstName || '',
    lastName: lastName || '',
    email: email || '',
    profilePicture: null,
    imageUrl: imageUrl || '',
  });
  const [prevPassword, setPrevPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [message, setMessage] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [hasPassword, setHasPassword] = useState(true);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwordToDeleteProfile, setPasswordToDeleteProfile] = useState('');
  const navigate = useNavigate();

  // Fetch user profile data
  useEffect(() => {
    if (user) {
      fetchUserProfile(user.uid);
    }
  }, [user]);

  // Function to fetch user profile data from firestore
  const fetchUserProfile = async (userId) => {
    try {
      const userDoc = await getUserDataFromFirestore(userId);
      if (userDoc.exists()) {
        const userData = userDoc.data();
        setUserProfile((prevProfile) => ({
          ...prevProfile,
          firstName: userData.firstName || '',
          lastName: userData.lastName || '',
          email: userData.email || '',
          imageUrl: userData.imageUrl || '',
        }));

        // Check the condition based on providerData for password existence
      const providerData = auth.currentUser.providerData;
      const hasPasswordProvider = providerData.some(
        (provider) => provider.providerId === 'password'
      );

      // Set the hasPassword state based on the existence of a password provider
      setHasPassword(hasPasswordProvider);
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
    }
  };

  // Function to retrieve data from firestore
  const getUserDataFromFirestore = async (userId) => {
    const userRef = doc(db, 'users', userId);
    return await getDoc(userRef);
  };

  // Handle input changes for form fields(profie picture)
  const handleInputChange = (e) => {
    const { name, type } = e.target;

    // If the input type is a file then handle profile picture changes
    if (type === 'file') {
      const file = e.target.files[0];
      const reader = new FileReader();

      // Set the userProfile state with the profile picture and imageUrl
      reader.onloadend = () => {
        setUserProfile((prevProfile) => ({
          ...prevProfile,
          profilePicture: file,
          imageUrl: reader.result,
        }));
        setSelectedImage(URL.createObjectURL(file));
      };

      // Read the file as a data URL
      if (file) {
        reader.readAsDataURL(file);
      }
    } else {
      // Update user profile for other input types
      setUserProfile((prevProfile) => ({
        ...prevProfile,
        [name]: e.target.value,
      }));
    }
    // Check if the input field is 'prevPassword' and has no password set
    if (name === 'prevPassword' && !hasPassword) {
      e.target.removeAttribute('disabled');
      return;
    }
  };

  // Handle removing profile picture
  const handleRemovePicture = () => {
    setSelectedImage(null);
    // If user exists and has a photoURL
    if (user && user.photoURL) {
      setUserProfile((prevProfile) => ({
        ...prevProfile,
        imageUrl: user.photoURL,
        profilePicture: null,
      }));
    } else {
      // If user deosn't have a photoURL
      setUserProfile((prevProfile) => ({
        ...prevProfile,
        imageUrl: '',
        profilePicture: null,
      }));
    }
  };

  // Handle updating user profile
  const handleUpdateProfile = async (e) => {
    e.preventDefault();

    try {

      const isPasswordValid = validatePassword(newPassword);

      // Fetch the user's data from Firestore
      const userDoc = await getUserDataFromFirestore(auth.currentUser.uid);

      // Check if the user has a password in Firestore
      const currentUser = auth.currentUser;
      const storedPassword = currentUser.providerData.some((provider)=>provider.providerId === 'password');

      // Update user profile information in Firestore
      if (userProfile.profilePicture) {
        await uploadProfilePicture(userProfile.profilePicture);
        const imageUrl = await getProfilePictureURL();
        // If profile picture is set
        await updateProfileFirebase(auth.currentUser, {
          displayName: `${userProfile.firstName} ${userProfile.lastName}`,
          photoURL: imageUrl,
        });
        await updateUserDataInFirestore({
          firstName: userProfile.firstName,
          lastName: userProfile.lastName,
          imageUrl: imageUrl,
        });

        // Update context API
        updateUser({
          email: userProfile.email,
          userID: user.uid,
          firstName: userProfile.firstName,
          lastName: userProfile.lastName,
          displayName: userProfile.displayName,
          imageUrl: userProfile.imageUrl,
        });

      } else {
        // If no new profile picture is set
        await updateProfileFirebase(auth.currentUser, {
          displayName: `${userProfile.firstName} ${userProfile.lastName}`,
          photoURL: userProfile.imageUrl || imageUrl,
        });

        await updateUserDataInFirestore({
          firstName: userProfile.firstName,
          lastName: userProfile.lastName,
          imageUrl: userProfile.imageUrl || imageUrl,
          password: newPassword,
        });

        updateUser({
          email: userProfile.email,
          userID: user.uid,
          firstName: userProfile.firstName,
          lastName: userProfile.lastName,
          displayName: userProfile.displayName,
          imageUrl: userProfile.imageUrl || imageUrl,
        });
      }


      // If the user doesn't have a password, set a new one
      if (!storedPassword) {
        await updatePassword(currentUser, newPassword);
      } else if (prevPassword) {
        // Reauthenticate user before updating the password
        try {
          const auth = getAuth();
          const credential = EmailAuthProvider.credential(currentUser.email, prevPassword);
          await reauthenticateWithCredential(auth.currentUser, credential);

          // Update user password
          await updatePassword(currentUser, newPassword);
        } catch (reauthError) {
          // Handle reauthentication error
          if (reauthError.code === 'auth/invalid-login-credentials'){
            displayMessage('Incorrect previous password. Please try again.');
          }else if(reauthError.code === 'auth/wrong-password') {
            // Display an error message when the provided previous password is incorrect
            displayMessage('Incorrect previous password. Please try again.');
          } else {
            // Handle other reauthentication errors
            displayMessage('Failed to reauthenticate. Please try again.');
          }
          
          return;
        }
      }

      // Validate the new password
      if (newPassword) {
        if(!isPasswordValid){
          return;
        } else{
          displayMessage('Profile updated successfully!');
        }
      } else{
        displayMessage('Profile updated successfully!');
      }
    } catch (error) {
      //console.error('Error updating profile:', error);
      displayMessage('Failed to update profile. Please try again.');
    }
  };

  // Update user data in firestore
  const updateUserDataInFirestore = async (data) => {
    const userRef = doc(db, 'users', auth.currentUser.uid);
    await updateDoc(userRef, data);
  };

  // Upload user profile picture to firebase storage
  const uploadProfilePicture = async (file) => {
    try {
      const storageRef = ref(storage, `profilePictures/${auth.currentUser.uid}`);
      await uploadBytes(storageRef, file);
      displayMessage('Profile updated successfully!');
    } catch (error) {
      //console.error('Error uploading profile picture:', error);
    }
  };

  // Get the URL of the user's profile picture
  const getProfilePictureURL = async () => {
    try {
      const storageRef = ref(storage, `profilePictures/${auth.currentUser.uid}`);
      const downloadURL = await getDownloadURL(storageRef);
      return downloadURL;
    } catch (error) {
      //console.error('Error getting profile picture URL:', error);
    }
  };

  // Handle cancel button click
  const handleCancel = () => {
    navigate('/Chat', {state: location.state});
  };

  // Delete all user data
  const handleDeleteProfile = async() => {
    if (hasPassword) {
      setShowPasswordModal(true);
    } else {
      // If user doesn't have a password, proceed with deletion directly
    const provider = new GoogleAuthProvider();
    try {
      await reauthenticateWithPopup(auth.currentUser, provider);
      await handleConfirmDeleteProfile('');
    } catch (error) {
      console.error('Error during reauthentication:', error);
      // Handle reauthentication error
      alert('Failed to reauthenticate using Google. Please try again.');
    }
    }
  };

  const handleClosePasswordModal = () => {
    setShowPasswordModal(false);
  };

  const handleConfirmDeleteProfile = async () => {
    try {
      const userId = auth.currentUser.uid;
      const profileImageURL = await getProfilePictureURL();
      // Fetch all conversation related to the user
      const querySnapshot = await getDocs(collection(db, 'conversations'));
      const conversationRefs = [];

      querySnapshot.forEach((doc) => {
        const documentId = doc.id;

        if (documentId.startsWith(userId)) {
          conversationRefs.push(doc.ref);
        }
      });
      // Delete all conversation related to the user
      for (const conversationRef of conversationRefs) {
        await deleteDoc(conversationRef);
      }
      // Check if user has profile picture and if has delete it
      if (profileImageURL) {
        const profileImageRef = storageRef(storage, `profilePictures/${auth.currentUser.uid}`);
        await deleteStorageObject(profileImageRef);
      }
      // Check if user has a password and if have ask a confirmation
      const password = passwordToDeleteProfile.trim();
      if (auth.currentUser.providerData[0]?.providerId === 'password') {
        if (!password) {
          alert('Please enter your password to confirm deletion.');
          return;
        }
        const credential = EmailAuthProvider.credential(auth.currentUser.email, password);
        await reauthenticateWithCredential(auth.currentUser, credential);
      }
      // Delete user data from firestore
      const userRef = doc(db, 'users', auth.currentUser.uid);
      await deleteDoc(userRef);

      await auth.currentUser.delete();

      navigate('/', { replace: true });
    } catch (error) {
      // Error handling for different deletion
      //console.error('Error deleting user data:', error);

      if(error.code === 'auth/invalid-login-credentials'){
        alert('Incorrect Password. Delection Canceled.')
      }
      else if(error.code === 'storage/object-not-found') {
        //console.warn('Profile picture not found in storage.');
      } 
      else if (error.code === 'auth/wrong-password') {
        alert('Incorrect password. Deletion canceled.');
      } 
      else {
        //console.error('Unhandled error:', error);
        //throw error;
      }
    } finally {
      //Close the password modal after deletion
      handleClosePasswordModal();
    }
  };

  // function for validate user enter password
  const validatePassword = (password) => {
    if (password){
      const passwordPolicy = "Password must include at leat one lower leter,upper letter, number & non-alphanumeric character(!@#$%&*).";

      if (password.length < 8 || password.length > 14) {
        displayMessage(`Password must be between 8 and 14 characters long. ${passwordPolicy}`);
          return false;
      }
    
      if (!/[a-z]/.test(password)) {
        displayMessage(passwordPolicy);
        return false;
      }
    
      if (!/[A-Z]/.test(password)) {
        displayMessage(passwordPolicy);
        return false;
      }
    
      if (!/[0-9]/.test(password)) {
        displayMessage(passwordPolicy);
        return false;
      }

      if (!/[^\w]/.test(password)) {
        displayMessage(passwordPolicy);
        return false;
      }
    
      return true;
    }
  };
  
  // Function to display alert message with a timeout
  const displayMessage = (message) => {
    setMessage(message);
  
    // Set a timeout to clear the message after 2 seconds
    const timeout = setTimeout(() => {
      setMessage('');
    }, 2000); 
  
    // Clear the timeout if the component unmounts or when a new message is displayed
    return () => clearTimeout(timeout);
  };


  return (
    <div className="container">
      <div className="profileHeader">
        <h1 className="text-primary mb-4 profileTitle">Edit Profile</h1>
      </div>
      <div className="profileDetails">
        <Col>
          <div className="profilePicture">
            <Col md={4}>
              <h3>Profile Picture</h3>
              {user && (selectedImage || user.photoURL || imageUrl || userIcon) && (
                <img
                  src={selectedImage || user.photoURL || imageUrl || userIcon}
                  alt="Profile"
                  className="img-fluid rounded-circle profileImage"
                />
              )}
              <Form.Group controlId="profilePicture">
                <Form.Label>Update Profile Picture</Form.Label>
                <Form.Control
                  type="file"
                  className="uploadPic"
                  name="profilePicture"
                  onChange={handleInputChange}
                />
              </Form.Group>
              {selectedImage && (
                <Button variant="danger" onClick={handleRemovePicture}>
                  Remove Picture
                </Button>
              )}
              {message && (
                <div className="alert alert-success">{message}</div>
              )}
            </Col>
          </div>
          <div className="text-center profileInfo">
            <Row>
              <Col md={4} className="personalInfo">
                <h3>Personal Info</h3>
                <Form onSubmit={handleUpdateProfile}>
                  <Form.Group controlId="firstName" className="firstName">
                    <Form.Label>First Name</Form.Label>
                    <Form.Control
                      type="text"
                      name="firstName"
                      value={userProfile.firstName}
                      onChange={handleInputChange}
                      required
                    />
                  </Form.Group>
                  <Form.Group controlId="lastName" className="lastName">
                    <Form.Label>Last Name</Form.Label>
                    <Form.Control
                      type="text"
                      name="lastName"
                      value={userProfile.lastName}
                      onChange={handleInputChange}
                      required
                    />
                  </Form.Group>
                  <Form.Group controlId="email" className="email">
                    <Form.Label>Email</Form.Label>
                    <Form.Control
                      type="email"
                      name="email"
                      value={userProfile.email}
                      onChange={handleInputChange}
                      required
                      disabled
                    />
                  </Form.Group>
                </Form>
              </Col>
              <Col md={4} className="updatePassword">
                <h3>Update Password</h3>
                <Form onSubmit={handleUpdateProfile}>
                  <Form.Group controlId="prevPassword" className="prevPassword">
                    <Form.Label>Previous Password</Form.Label>
                    <Form.Control
                      type="password"
                      name="prevPassword"
                      value={prevPassword}
                      onChange={(e) => setPrevPassword(e.target.value)}
                      required
                      disabled={!hasPassword}
                    />
                  </Form.Group>
                  <Form.Group controlId="newPassword" className="newPassword">
                    <Form.Label>New Password</Form.Label>
                    <Form.Control
                      type="password"
                      name="newPassword"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      required
                    />
                  </Form.Group>
                </Form>
              </Col>
            </Row>
            <div className="buttons-container">
              <Button
                type="submit"
                variant="primary"
                className="submitBtn"
                onClick={handleUpdateProfile}
              >
                Save Changes
              </Button>
              <Button type="button" variant="secondary" className="cancelBtn" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="button" variant="danger" className="deleteBtn" onClick={handleDeleteProfile}>
                Delete Profile
              </Button>
            </div>
          </div>
        </Col>
      </div>
      <Modal show={showPasswordModal} onHide={handleClosePasswordModal}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm Deletion</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group controlId="formPassword">
              <Form.Label>Enter your password to confirm deletion:</Form.Label>
              <Form.Control
                type="password"
                placeholder="Password"
                value={passwordToDeleteProfile}
                onChange={(e) => setPasswordToDeleteProfile(e.target.value)}
              />
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleClosePasswordModal}>
            Close
          </Button>
          <Button variant="danger" onClick={handleConfirmDeleteProfile}>
            Confirm Deletion
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
