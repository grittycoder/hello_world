// Simplified logic for the mobile toggle
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
