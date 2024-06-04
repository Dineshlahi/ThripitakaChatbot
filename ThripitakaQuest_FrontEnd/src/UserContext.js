import { createContext, useContext, useState, useEffect } from 'react';

const UserContext = createContext();

export const UserProvider = ({ children }) => {
  // Try to get user data from localStorage on initial load
  const initialUserData = JSON.parse(localStorage.getItem('userData')) || '';
  const [users, setUsers] = useState(initialUserData);

  useEffect(() => {
    // Save user data to localStorage whenever it changes
    localStorage.setItem('userData', JSON.stringify(users));
  }, [users]);

  const updateUser = (userData) => {
    console.log('Updating user data:', userData);
    setUsers(userData);
    console.log('User data updated:', users);
  };

  return (
    <UserContext.Provider value={{ users, updateUser }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUserContext = () => {
  return useContext(UserContext);
};
