import React, { useState } from 'react';

const Register = () => {
  const [role, setRole] = useState('client'); // Default to client

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4">
      <div className="max-w-md w-full space-y-8 bg-white p-10 rounded-xl shadow-md">
        <h2 className="text-center text-3xl font-extrabold text-gray-900">
          Create your account
        </h2>
        
        {/* Role Selector */}
        <div className="flex gap-4">
          <button 
            onClick={() => setRole('client')}
            className={`flex-1 p-4 border rounded-lg ${role === 'client' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}
          >
            <span className="block text-2xl">🏠</span>
            <span className="font-bold">I need a clean</span>
          </button>
          <button 
            onClick={() => setRole('cleaner')}
            className={`flex-1 p-4 border rounded-lg ${role === 'cleaner' ? 'border-green-500 bg-green-50' : 'border-gray-200'}`}
          >
            <span className="block text-2xl">🧼</span>
            <span className="font-bold">I want to work</span>
          </button>
        </div>

        <form className="mt-8 space-y-4">
          <input type="email" placeholder="Email Address" className="w-full p-3 border rounded-md" />
          <input type="password" placeholder="Create Password" className="w-full p-3 border rounded-md" />
          
          {role === 'cleaner' && (
            <textarea 
              placeholder="Tell us about your background..." 
              className="w-full p-3 border rounded-md"
            />
          )}

          <button className="w-full py-3 bg-indigo-600 text-white rounded-md font-bold">
            Sign Up as {role === 'client' ? 'Client' : 'Cleaner'}
          </button>
        </form>
      </div>
    </div>
  );
};
