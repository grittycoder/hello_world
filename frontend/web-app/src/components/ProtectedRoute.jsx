// src/components/ProtectedRoute.jsx

import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children, allowedRoles, user }) => {
  // 1. Check if logged in
  if (!user || !user.token) {
    return <Navigate to="/login" replace />;
  }

  // 2. Check if the role is authorized for this specific path
  if (!allowedRoles.includes(user.role)) {
    // Send them to a "safe" default page if they don't belong here
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
};

export default ProtectedRoute;
