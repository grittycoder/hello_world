// shared/validation.js
export const validateRegistration = (data) => {
  const errors = {};
  if (!data.email.includes('@')) errors.email = "Invalid email address";
  if (data.password.length < 8) errors.password = "Password must be 8+ characters";
  
  // Cleaner-specific requirement
  if (data.role === 'cleaner' && !data.experience) {
    errors.experience = "Please tell us about your cleaning experience";
  }
  
  return errors;
};
