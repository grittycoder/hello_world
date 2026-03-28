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

// shared/registrationSchema.js
import * as Yup from 'yup';

export const registrationSchema = Yup.object().shape({
  email: Yup.string().email('Invalid email').required('Email is required'),
  password: Yup.string().min(8, 'Password must be at least 8 characters').required('Password is required'),
  role: Yup.string().oneOf(['customer', 'cleaner']).required('Role is required'),
  experience: Yup.string().when('role', {
    is: 'cleaner',
    then: Yup.string().required('Please tell us about your cleaning experience'),
    otherwise: Yup.string().notRequired()
  })
});











