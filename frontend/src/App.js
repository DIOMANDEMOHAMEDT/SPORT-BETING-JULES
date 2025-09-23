import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import History from './components/History';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow-md">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex-shrink-0 text-2xl font-bold text-gray-800">
                BettingApp
              </div>
              <div className="flex items-baseline space-x-4">
                <Link to="/" className="text-gray-600 hover:bg-gray-200 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
                  Dashboard
                </Link>
                <Link to="/history" className="text-gray-600 hover:bg-gray-200 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
                  History
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main>
          <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/history" element={<History />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
