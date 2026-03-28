// ClientDashboard.jsx

import React from 'react';

const ClientDashboard = ({ activeJob }) => {

  // Logic to calculate percentage based on completed tasks

  const completedCount = activeJob.tasks.filter(t => t.completed).length;
  const progressPercent = (completedCount / activeJob.tasks.length) * 100;

  return (
    <div className="p-6 max-w-2xl mx-auto bg-white shadow-lg rounded-2xl">
      <h2 className="text-2xl font-bold mb-4">Live Service Update</h2>
      
      {/* The Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-4 mb-6">
        <div 
          className="bg-green-500 h-4 rounded-full transition-all duration-500" 
          style={{ width: `${progressPercent}%` }}
        ></div>
      </div>

      <div className="space-y-4">
        {activeJob.tasks.map((task, index) => (
          <div key={index} className="flex items-center gap-3">
            <span className={task.completed ? "text-green-500" : "text-gray-300"}>
              {task.completed ? "●" : "○"}
            </span>
            <span className={task.completed ? "line-through text-gray-400" : "text-gray-700"}>
              {task.name}
            </span>
          </div>
        ))}
      </div>
      
      <p className="mt-6 text-sm text-center text-gray-500 italic">
        "Your cleaner is currently ... the ..."
      </p>
    </div>
  );
};