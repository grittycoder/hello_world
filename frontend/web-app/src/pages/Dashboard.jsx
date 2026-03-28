// DashBoard.jsx


import React from 'react';
import { Link } from 'react-router-dom';
const Dashboard = () => {
  return (
    <div className="p-6 max-w-2xl mx-auto bg-white shadow-lg rounded-2xl">
      <h1 className="text-3xl font-bold mb-6">Welcome to Your Dashboard</h1>
      <p className="text-gray-700 mb-4">Here you can manage your services and view updates.</p>
        <div className="space-y-4">
            <Link to="/client-dashboard" className="block w-full text-center py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-300">
                View Live Service Update
            </Link>
            <Link to="/settings" className="block w-full text-center py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors duration-300">
                Account Settings    
            </Link>
        </div>
    </div>
  );
}