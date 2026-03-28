// Simplified logic for the mobile toggle between Client and Cleaner registration

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

const RegisterScreen = () => {
  const [role, setRole] = useState('client');


return (
  <View style={styles.container}>
    <Text style={styles.title}>Join SparkleSquad</Text>
    
    <View style={styles.toggleContainer}>
      <TouchableOpacity 
        style={[styles.toggleBtn, role === 'client' && styles.active]} 
        onPress={() => setRole('client')}
      >
        <Text>Client</Text>
      </TouchableOpacity>
      <TouchableOpacity 
        style={[styles.toggleBtn, role === 'cleaner' && styles.active]} 
        onPress={() => setRole('cleaner')}
      >
        <Text>Cleaner</Text>
      </TouchableOpacity>
    </View>

    {/* Rest of the form inputs... */}
  </View>
);

};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  toggleContainer: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  toggleBtn: {
    flex: 1,
    padding: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
  },
  active: {
    backgroundColor: '#007BFF',
    borderColor: '#007BFF',
  },
});
export default RegisterScreen;

