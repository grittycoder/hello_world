# Booking_Screen
const renderServiceItem = ({ item }) => (
  <TouchableOpacity 
    onPress={() => navigation.navigate('Booking', { service: item })}
  >
    {/* card content */}
  </TouchableOpacity>
);





import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Alert } from 'react-native';

const BookingScreen = ({ route, navigation }) => {
  // Assume the service was passed via navigation
  const service = route?.params?.service || { title: 'Standard Consultation', price: '$50' };
  
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [selectedTime, setSelectedTime] = useState('10:00 AM');

  const handleConfirmBooking = () => {
    // Navigate to Payment Processing next
    navigation.navigate('Payment', { 
      service, 
      date: selectedDate.toDateString(), 
      time: selectedTime 
    });
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.summaryCard}>
        <Text style={styles.label}>Service Selected</Text>
        <Text style={styles.title}>{service.title}</Text>
        <Text style={styles.price}>{service.price}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select Date & Time</Text>
        
        <TouchableOpacity style={styles.selectorButton} onPress={() => {/* Trigger Date Picker */}}>
          <Text style={styles.selectorText}>{selectedDate.toDateString()}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.selectorButton} onPress={() => {/* Trigger Time Picker */}}>
          <Text style={styles.selectorText}>{selectedTime}</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.confirmButton} onPress={handleConfirmBooking}>
        <Text style={styles.confirmButtonText}>Proceed to Payment</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f9f9f9', padding: 20 },
  summaryCard: { backgroundColor: '#fff', padding: 20, borderRadius: 12, marginBottom: 20, elevation: 2 },
  label: { color: '#888', fontSize: 12, textTransform: 'uppercase', marginBottom: 4 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#333' },
  price: { fontSize: 18, color: '#007AFF', marginTop: 4 },
  section: { marginBottom: 30 },
  sectionTitle: { fontSize: 16, fontWeight: '600', marginBottom: 12, color: '#444' },
  selectorButton: { backgroundColor: '#fff', padding: 16, borderRadius: 8, borderWidth: 1, borderColor: '#ddd', marginBottom: 12 },
  selectorText: { fontSize: 16, color: '#333' },
  confirmButton: { backgroundColor: '#007AFF', padding: 18, borderRadius: 8, alignItems: 'center' },
  confirmButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
});


const calculateTotal = (service, rooms, baths, selectedAddons) => {
  const base = service.basePrice;
  const roomCost = rooms * service.modifiers.pricePerBedroom;
  const bathCost = baths * service.modifiers.pricePerBathroom;
  const addonCost = selectedAddons.reduce((sum, addon) => sum + addon.price, 0);
  
  return base + roomCost + bathCost + addonCost;
};

export default BookingScreen;
