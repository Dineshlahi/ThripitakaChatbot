import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SignIn from './Pages/SignIn';
import SignUp from './Pages/SignUp';
import Chat from './Pages/Chat';
import Profile from './Pages/Profile';
import PasswordReset from './Pages/PasswordReset';
import NewPassword from './Pages/NewPassword';

function App() {
  return (
    <Router>
      <div>
        <Routes>
          <Route exact path="/" element={<SignIn />} />
          <Route path="/SignUp" element={<SignUp/>}/>
          <Route path="/Chat" element={<Chat/>}/>
          <Route path="/Profile" element={<Profile/>}/>
          <Route path="/PasswordReset" element={<PasswordReset/>}/>
          <Route path="/NewPassword" element={<NewPassword/>} />
        </Routes>
      </div>
    </Router>
  )
}

export default App;
