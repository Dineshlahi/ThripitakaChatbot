import { Image, Button } from "react-bootstrap";
import "./SideBar.css"
import logo from "../assets/ThripitakaQuestLogo.png";
import deleteIcon from "../assets/delete.png";
import userIcon from "../assets/userIcon.png";
import addBtnLogo from "../assets/add-30.png";
import message from "../assets/message.svg";
import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from "react";
import { db } from '../Pages/firebase';
import { getAuth, signOut } from 'firebase/auth';
import { collection, query, where, doc, getDocs, startsWith, deleteDoc } from 'firebase/firestore';
import ProfileOptionsPopup from '../Components/ProfileOption';
import { useUserContext } from '../UserContext';


export default function (props) {

  // Use usestate to check whether profile option modal is display or not
  const [showProfileOptions, setShowProfileOptions] = useState(false);
  // Use usestate to check whether profile option modal is selected or not
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  // Use usestate to retrieve data from the conversation documents in firebse
  const [conversations, setConversations] = useState([]);
  // Use usestate to check whether use has selected a query button
  const [queryButtonClicked, setQueryButtonClicked] = useState(false);
  // Use usestate to retrieve conversationId in the firebase 
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const [keyForRerender, setKeyForRerender] = useState(0);

  // Use context API to pass data
  const { users } = useUserContext();
  const { email, userID, firstName, lastName, displayName, imageUrl } = users;
  const location = useLocation();
  const navigate = useNavigate();


  // Function for fetch conversation documents from the firebase
  const fetchConversations = async () => {
    try {
      const userId = userID;
      // Retrieving documents from the 'conversations' collection in the firestore
      const querySnapshot = await getDocs(collection(db, 'conversations'));
      const conversationData = [];
      // Iterate through each document in the querySnapshot
      querySnapshot.forEach((doc) => {
        const documentId = doc.id;
        // If the document Id start with user's ID retrieving document data and push to the array
        // Conversation documents naming convention is 'userID_randomly digits'
        if (documentId.startsWith(userId)) {
          const data = doc.data();
          conversationData.push({ id: documentId, ...data });
        }
      });
      // Set conversation using the retireved data
      setConversations(conversationData);
    } // Handeling if any errors occur
    catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  // Fetch existing conversations
  useEffect(() => {
    fetchConversations();
  }, [userID]);


  // Add an onClick handler to the "New Chat" button to trigger the database update
  const handleNewChatClick = async () => {
    props.onNewChatClick();
    setQueryButtonClicked(false);
    // After creating a new chat, update the conversation list
    await fetchConversations();
  };

  // Function to handle the query button selection done by user
  const handleQueryButtonClick = (conversationId) => {
    props.onQueryButtonClick(conversationId);
    setQueryButtonClicked(true);
  };


  // Pass a function to reset the current conversation to mainbar
  const resetCurrentConversation = () => {
    props.resetCurrentConversation();
  }

  // Function to delete a conversation from the database
  const handleDeleteButtonClick = async (conversationId) => {
    try {
      // Implement logic to delete the conversation document
      await deleteConversation(conversationId);

      // Reset the selected conversation ID to null
      setSelectedConversationId(null);

      // Update the conversation list
      await fetchConversations();

      // Call the function to reset the current conversation
      resetCurrentConversation();
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };

  // Function for the delete conversation
  const deleteConversation = async (conversationId) => {
    try {
      // Implement logic to delete the conversation document
      const conversationRef = doc(db, 'conversations', conversationId);
      await deleteDoc(conversationRef);
      // Reset the selectedConversationId
      setSelectedConversationId(null);
    } catch (error) {
      console.error('Error deleting conversation document:', error);
    }
  };

  // Function to handle the click event to display profile options
  const handleProfileOptionsClick = () => {
    setShowProfileOptions(true);
  };

  // Function to handle the close event to display profile options
  const handleProfileOptionsClose = () => {
    setShowProfileOptions(false);
  };

  // Function to handle the click profile settings button in modal
  const handleSettingsClick = () => {
    navigate('/Profile', { state: location.state });
  };

  // Function to close the settings modal
  const handleSettingsClose = () => {
    setShowSettingsModal(false);
  };

  // Function to handle the logout button in modal
  const handleLogoutClick = async () => {
    try {
      // Get the authentication instance
      const auth = getAuth();
      await signOut(auth);

      // Reset local state
      setConversations([]);
      setShowProfileOptions(false);

      navigate('/', { replace: true })

      // Clear local storage or session storage (optional)
      localStorage.clear();
      sessionStorage.clear();

      // Reload the page to ensure a clean state after logout
      window.location.reload();
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  return (
    <div className="sideBar">
      <div className="upperSide">
        <div className="upperSideTop">
          <Image src={logo} rounded className="logo" />

        </div>

        <button variant="primary" className="midButton"><Image src={addBtnLogo} rounded className="newChatBtnImg" onClick={handleNewChatClick} /> New Chat</button>{' '}

        <div className="upperSideBottom">
          {conversations.length > 0 ? (
            conversations.map((conversation) => (
              <div key={conversation.id} className="queryButtonContainer">
                <Button
                  variant="primary"
                  className="queryButton"
                  onClick={() => handleQueryButtonClick(conversation.id)}
                  onMouseEnter={() => setSelectedConversationId(conversation.id)}
                  onMouseLeave={() => setSelectedConversationId(null)}
                >
                  <Image src={message} rounded className="queryBtnImg" />
                  {conversation.messages && conversation.messages.length > 0 && conversation.messages[0].text && (
                    <span>{conversation.messages[0].text}</span>
                  )}
                  {/* Delete icon button */}
                  {selectedConversationId === conversation.id && (
                    <Image
                      src={deleteIcon}
                      rounded
                      className="deleteBtnImg"
                      onClick={() => {
                        handleDeleteButtonClick(conversation.id);
                        setSelectedConversationId(conversation.id);
                      }}
                    />
                  )}
                </Button>
              </div>
            ))
          ) : (
            <></>
          )}
        </div>
      </div>
      <div className="lowerSide">
        <div className="listItems">
          <button className="profileBtn" onClick={handleProfileOptionsClick}>
            <Image src={imageUrl ? imageUrl : userIcon} roundedCircle className="listItemsImg" /> {displayName || `${firstName} ${lastName}`}
          </button>
        </div>
      </div>
      {/* Profile Options Popup */}
      <ProfileOptionsPopup
        show={showProfileOptions}
        onHide={handleProfileOptionsClose}
        onSettingsClick={handleSettingsClick}
        onLogoutClick={handleLogoutClick}
      />
    </div>
  );
}


