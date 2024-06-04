// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
import {getStorage} from "firebase/storage";

// Web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyC-QJsi50zjHn3vLYl1c8NXg9pW3usebCA",
  authDomain: "thripitaka-ai.firebaseapp.com",
  projectId: "thripitaka-ai",
  storageBucket: "thripitaka-ai.appspot.com",
  messagingSenderId: "15378879590",
  appId: "1:15378879590:web:a3324d0faeb031c3b909a7",
  measurementId: "G-FH0RCCG0SX"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const storage = getStorage(app);
// Initialize Firestore
const db = getFirestore(app);

export { db, app, auth, analytics, storage};