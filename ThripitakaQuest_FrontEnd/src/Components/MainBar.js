import { Image, Button } from "react-bootstrap";
import sendBtn from "../assets/send.svg";
import userIcon from "../assets/userIcon.png";
import logo from "../assets/ThripitakaQuestLogo.png";
import { useRef, useEffect, useState } from "react";
import { useLocation } from 'react-router-dom';
import { doc, collection, updateDoc, arrayUnion, getDoc, getDocs, setDoc } from 'firebase/firestore';
import { db } from '../Pages/firebase';
import axios from 'axios';
import { useUserContext } from '../UserContext';
import "./MainBar.css";

export default function MainBar(props) {
  // Destructuring data from the useUserContext hook
  const { users } = useUserContext();
  const { userID, imageUrl } = users;
  //Creating a reference to an element to scroll to the bottom of the chat window
  const msgEnd = useRef(null);

  // State variables using useState hook
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([{
    text: 'Hi, I am Thripitaka Quest. How can I help you...',
    isBot: true,
  }]);
  const [conversationID, setConversationID] = useState(generateConversationID());
  const [loadingOldConversation, setLoadingOldConversation] = useState(false);
  const [isNewConversation, setNewConversation] = useState(false);

  // Function to start a new conversation
  const startNewConversation = () => {
    // Generate a new conversation ID
    const newConversationID = generateConversationID();
    setConversationID(newConversationID);

    // Clear the chat history
    setMessages([]);
    setNewConversation(true);
  };

  // Effect to scroll to the end fo the chat window 
  useEffect(() => {
    msgEnd.current.scrollIntoView();
  }, [messages]);

  // Handle a new conversation if requested
  useEffect(() => {
    if (props.isNewConversation) {
      setLoadingOldConversation(false);
      // Clear the chat history if a new conversation is requested
      setMessages([]);
      setConversationID(generateConversationID());
      // Reset isNewConversation to false after successfully creating a new conversation
      props.resetNewConversation();
    }
  }, [props.isNewConversation]);


  // Function to create a conversation if it doesn't exist
  const createConversationIfNotExists = async (conversationRef) => {
    const docSnapshot = await getDoc(conversationRef);

    if (!docSnapshot.exists()) {
      await setDoc(conversationRef, { messages: [] });
    }
  };

  //Load past messages when an old conversation is selected
  useEffect(() => {
    if (props.selectedConversation || props.queryButtonClicked) {
      const selectedId = props.selectedConversation;
      loadPastMessages(selectedId);
      setLoadingOldConversation(true);
    }
  }, [props.selectedConversation, props.queryButtonClicked]);

  // Function to lod past messages for a conversation
  const loadPastMessages = async (conversationId) => {
    try {
      // Create a refrence to the conversation document
      const conversationRef = doc(db, 'conversations', conversationId);
      // Retrieving the document snapshot from firebase
      const docSnapshot = await getDoc(conversationRef);

      // Check document exists and if exist, extract conversation data
      if (docSnapshot.exists()) {
        const conversationData = docSnapshot.data();
        if (conversationData && conversationData.messages) {
          setMessages([...conversationData.messages]);
          // Scroll to the bottom of the chat window after loading messages
          msgEnd.current.scrollIntoView({ behavior: 'smooth' });
        }
      }
    } catch (error) {
      console.error('Error loading past messages:', error);
    }
  };


////////////////////////////////////////


  // Send conversation data into database
  const handleSend = async () => {
    const text = input;
    if (text === undefined || text.trim() === '') {
      console.error('Error: Text is undefined of empty');
      return;
    }
    setInput('');

    // Create user message object
    const userMessage = { text: text, sender: 'user', isBot: false, timestamp: new Date() };
    setMessages([...messages, userMessage]);
    let botMessage; 
    try {
      const response = await axios.post('http://127.0.0.1:5000/chatbot/ask', { query: text });
      
      // Store the response data in a variable
      const botResponseData = response.data;
    
      // Create a bot message using the response data
        botMessage = {
        text: botResponseData.response,
        sender: 'bot',
        isBot: true,
        timestamp: new Date()
      };
    
      // Update state with the new bot message
      setMessages([...messages, botMessage]);
    } catch (error) {
      console.log(error);
    };

      //const botMessage = { text: botResponse.data.message, sender: 'bot', isBot: true, timestamp: new Date() };

      // update the conversation in firestore based on the conversation status
      if (loadingOldConversation) {
        // If loading an old conversation, update the existing conversation
        const conversationRef = doc(collection(db, 'conversations'), props.selectedConversation);
        setMessages([...messages, userMessage, botMessage]);
        updateFirebaseWithMessages(conversationRef, userMessage, botMessage);
      } else {
        // If starting a new conversation, create a new conversation
        const conversationRef = doc(collection(db, 'conversations'), conversationID);
        await createConversationIfNotExists(conversationRef);
        setMessages([...messages, userMessage, botMessage]);
        updateFirebaseWithMessages(conversationRef, userMessage, botMessage);
      }

  };



  // Function for update the firebase with messages
  const updateFirebaseWithMessages = async (conversationRef, userMessage, botMessage) => {
    try {
      await updateDoc(conversationRef, {
        messages: arrayUnion(userMessage, botMessage)
      });
    } catch (error) {
      console.error('Error updating Firestore:', error);
    }
  };

  // Function for genereate unique conversation document identifier
  function generateConversationID() {
    const uniqueIdentifier = generateUniqueIdentifier();
    return `${userID}_${uniqueIdentifier}`;
  }

  // Function for generate a random 6-character alphanumeric string:
  function generateUniqueIdentifier() {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let uniqueIdentifier = '';
    for (let i = 0; i < 6; i++) {
      const randomIndex = Math.floor(Math.random() * characters.length);
      uniqueIdentifier += characters.charAt(randomIndex);
    }
    return uniqueIdentifier;
  }

  // Function for handle enter key option in user inputbar
  const handleEnter = async (e) => {
    if (e.key == 'Enter') await handleSend();
  }


  const refreshAndStartNewConversation = () => {
    setMessages([]);
    setConversationID(generateConversationID());
    setNewConversation(true);
  };

  return ( 
    <div className="mainBar">
      <div className="chatWindow">
        {messages.map((message, i) =>
          <div key={i} className={message.isBot ? "chat bot" : "chat user"}>
            <Image src={message.isBot ? logo : imageUrl ? imageUrl : userIcon} rounded className="profilePic" />
            <p className="questionParagraph"> {message.text}</p>
          </div>
        )}
        <div ref={msgEnd} />
      </div>
      <div className="chatFooter">
        <div className="inputBar">
          <input type="text" placeholder="Send a message"
            className="msgHolder" value={input} onKeyDown={handleEnter}
            onChange={(e) => { setInput(e.target.value) }} />
          <Button variant="primary" className="sendBtn" onClick={handleSend}><Image src={sendBtn} rounded className="sendBtnImg" /> </Button>{' '}
        </div>
        <p>This is the first version of Thripitaka Quest</p>
      </div>
    </div>
  );
}
