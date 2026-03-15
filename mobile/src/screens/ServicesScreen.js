# Service Me, Service You

import React from 'react';
import { 
  View, 
  Text, 
  FlatList, 
  TouchableOpacity, 
  Image, 
  StyleSheet, 
  SafeAreaView 
} from 'react-native';

// Temporary seed data for UI development
import React from 'react';
import { FlatList } from 'react-native';
// Import the seed data
import seedData from '../data/seedData.json'; 

const ServicesScreen = () => {
  // Use the services array from the JSON file
  const services = seedData.services;

  return (
    <FlatList
      data={services}
      keyExtractor={(item) => item.id}
      renderItem={({ item }) => (
        // Your existing card component here
        <ServiceCard title={item.title} price={item.price} />
      )}
    />
  );
};


const ServicesScreen = ({ navigation }) => {

  const renderServiceItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.card}
      onPress={() => console.log(`Selected: ${item.title}`)}
    >
      <View style={styles.cardContent}>
        <View>
          <Text style={styles.serviceTitle}>{item.title}</Text>
          <Text style={styles.serviceDetails}>{item.duration} • {item.price}</Text>
        </View>
        <View style={styles.arrowContainer}>
          <Text style={styles.arrow}>→</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Our Services</Text>
        <Text style={styles.headerSubtitle}>Select a service to book an appointment</Text>
      </View>

      <FlatList
        data={MOCK_SERVICES}
        renderItem={renderServiceItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  listContainer: {
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    // Shadow for iOS
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    // Elevation for Android
    elevation: 3,
    borderWidth: Platform.OS === 'ios' ? 0 : 1,
    borderColor: '#eee',
  },
  cardContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  serviceTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  serviceDetails: {
    fontSize: 14,
    color: '#888',
    marginTop: 4,
  },
  arrowContainer: {
    backgroundColor: '#f0f7ff',
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  arrow: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
});

export default ServicesScreen;
