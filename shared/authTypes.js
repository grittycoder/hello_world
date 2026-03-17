// shared/authTypes.js
export const ROLES = {
  CLIENT: 'client',
  CLEANER: 'cleaner',
  ADMIN: 'admin'
};

// This logic will be used by both React (Web) and React Native (Mobile)
export const isAuthenticated = (user) => !!user && !!user.token;
