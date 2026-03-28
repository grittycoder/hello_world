// shared/authTypes.js

export const ROLES = {
  CLIENT: 'client',
  CLEANER: 'cleaner',
  ADMIN: 'admin'
};

// This logic will be used by both React (Web) and React Native (Mobile)

export const isAuthenticated = (user) => !!user && !!user.token;

// Example Server Logic

const token = jwt.sign(
  { id: user._id, role: user.role }, 
  process.env.JWT_SECRET, 
  { expiresIn: '7d' }
);

// Example Client Logic

import jwtDecode from 'jwt-decode';

const token = localStorage.getItem('token');
if (token) {
  const decoded = jwtDecode(token);
  const user = {
    id: decoded.id,
    role: decoded.role,
    token
  };
  // Set user in state or context
  setUser(user);
}
