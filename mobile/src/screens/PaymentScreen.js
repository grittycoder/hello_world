// Payment integration using Stripe's React Native SDK

import { useStripe } from '@stripe/stripe-react-native';

const PaymentScreen = () => {
  const { initPaymentSheet, presentPaymentSheet } = useStripe();

  const initializePayment = async () => {
    // 1. Get PaymentIntent and EphemeralKey from your backend
    const { paymentIntent, customer } = await fetchPaymentParams();

    // 2. Initialize the native UI
    const { error } = await initPaymentSheet({
      customerId: customer,
      paymentIntentClientSecret: paymentIntent,
      merchantDisplayName: 'SparkleSquad Inc.',
    });

    if (!error) {
      // 3. Open the slide-up menu
      const { error: paymentError } = await presentPaymentSheet();
      if (!paymentError) alert("Payment authorized!");
    }
  };

  return (
    <TouchableOpacity style={styles.payBtn} onPress={initializePayment}>
      <Text>Finish Booking</Text>
    </TouchableOpacity>
  );
};

import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';

const PaymentScreen = ({ route }) => {
  const { service } = route.params;
  const [loading, setLoading] = useState(false);

  const processPayment = () => {
    setLoading(true);
    // Simulate a network request
    setTimeout(() => {
      setLoading(false);
      alert("Payment Successful!");
    }, 2000);
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.totalSection}>
        <Text style={styles.totalLabel}>Total Amount</Text>
        <Text style={styles.totalAmount}>{service.price}</Text>
      </View>

      <View style={styles.cardForm}>
        <TextInput style={styles.input} placeholder="Cardholder Name" />
        <TextInput style={styles.input} placeholder="Card Number" keyboardType="numeric" />
        
        <View style={styles.row}>
          <TextInput style={[styles.input, { flex: 1, marginRight: 10 }]} placeholder="MM/YY" />
          <TextInput style={[styles.input, { flex: 1 }]} placeholder="CVC" keyboardType="numeric" secureTextEntry />
        </View>
      </View>

      <TouchableOpacity 
        style={[styles.payButton, loading && { backgroundColor: '#ccc' }]} 
        onPress={processPayment}
        disabled={loading}
      >
        <Text style={styles.payButtonText}>
          {loading ? "Processing..." : `Pay ${service.price}`}
        </Text>
      </TouchableOpacity>
      
      <Text style={styles.secureText}>🔒 Secure SSL Encrypted Payment</Text>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', padding: 24, justifyContent: 'center' },
  totalSection: { alignItems: 'center', marginBottom: 40 },
  totalLabel: { fontSize: 14, color: '#666' },
  totalAmount: { fontSize: 36, fontWeight: 'bold', color: '#1a1a1a' },
  cardForm: { marginBottom: 20 },
  input: { backgroundColor: '#f5f5f5', padding: 15, borderRadius: 8, marginBottom: 15, fontSize: 16 },
  row: { flexDirection: 'row' },
  payButton: { backgroundColor: '#28a745', padding: 18, borderRadius: 8, alignItems: 'center', marginTop: 10 },
  payButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  secureText: { textAlign: 'center', marginTop: 20, color: '#aaa', fontSize: 12 },
});

export default PaymentScreen;
