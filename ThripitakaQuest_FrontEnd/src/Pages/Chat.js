import React from "react";
import SideBar from "../Components/SideBar";
import MainBar from "../Components/MainBar";
import { useState } from "react";
import './Chat.css'

export default function Chat(){

  // Use usestate to check whether user click the new chat button
  const [isNewConversation, setNewConversation] = useState(false);
  // Use usestate to retrieve the id of the user selected conversation id
  const [selectedConversation, setSelectedConversation] = useState(null);
  // Use usestate to check whether user click a query button
  const [queryButtonClicked, setQueryButtonClicked] = useState(false);
  // Use usestate to check whether user choose and load an his previous conversation
  const [loadingOldConversation, setLoadingOldConversation] = useState(false);

  // Change the usestates when the "New Chat" button is clicked
  const handleNewChatClick = () => {
    setNewConversation(true);
    setQueryButtonClicked(false);
    setLoadingOldConversation(false);
  };

  // Define a function to reset usesate settings to false
  const resetNewConversation = () => {
    setNewConversation(false);
    setQueryButtonClicked(false);
    setLoadingOldConversation(false);
  };

  // Handle the selection of a conversation in the Sidebar
  const handleQueryButtonClick = (conversationId) => {
    setSelectedConversation(conversationId);
    setQueryButtonClicked(true);
    setLoadingOldConversation(true);
  };

  // Function to reset the current conversation in MainBar
  const resetCurrentConversation = () => {
    setNewConversation(true);
    setSelectedConversation(null);
    setQueryButtonClicked(false);
    setLoadingOldConversation(false);
  };

  return (
    <div className="homeChat">
      <SideBar onNewChatClick={handleNewChatClick} onQueryButtonClick={handleQueryButtonClick} 
                            resetCurrentConversation={resetCurrentConversation} />
      <MainBar isNewConversation={isNewConversation} resetNewConversation={resetNewConversation} selectedConversation={selectedConversation}
      queryButtonClicked={queryButtonClicked} loadingOldConversation={loadingOldConversation}/>
    </div>
  );
}