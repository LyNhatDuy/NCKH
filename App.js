import React, { useState } from 'react';
import logo from './logo.svg';
import WritingExercise from './components/WritingExercise';
import GrammarTest from './components/GrammarTest';
import './App.css';

function App() {
  // State để chuyển đổi giữa giao diện mặc định và giao diện học tiếng Anh
  const [showLearningMode, setShowLearningMode] = useState(false);

  return (
    <div className="App">
      {showLearningMode ? (
        // Giao diện "Học Tiếng Anh"
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-2xl">
            <h1 className="text-2xl font-bold mb-4 text-center">Học Tiếng Anh</h1>
            <GrammarTest />
            <WritingExercise />
            <button
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={() => setShowLearningMode(false)}
            >
              Quay về giao diện mặc định
            </button>
          </div>
        </div>
      ) : (
        // Giao diện mặc định của Create React App
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <p>
            Edit <code>src/App.js</code> and save to reload.
          </p>
          <a
            className="App-link"
            href="https://reactjs.org"
            target="_blank"
            rel="noopener noreferrer"
          >
            Learn React
          </a>
          <button
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            onClick={() => setShowLearningMode(true)}
          >
            Chuyển sang chế độ học
          </button>
        </header>
      )}
    </div>
  );
}

export default App;