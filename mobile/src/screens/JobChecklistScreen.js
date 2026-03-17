import React, { useState } from 'react';
import { View, Text, TouchableOpacity, FlatList, StyleSheet } from 'react-native';
import jobSchema from '../../../shared/jobSchema.json';

const JobChecklistScreen = ({ route }) => {
  const { serviceId = 'clean-std' } = route.params || {};
  const [tasks, setTasks] = useState(
    jobSchema.taskTemplates[serviceId].map(t => ({ name: t, completed: false }))
  );

  const toggleTask = (index) => {
    const newTasks = [...tasks];
    newTasks[index].completed = !newTasks[index].completed;
    setTasks(newTasks);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Service in Progress</Text>
      <FlatList
        data={tasks}
        renderItem={({ item, index }) => (
          <TouchableOpacity 
            style={[styles.taskItem, item.completed && styles.taskCompleted]} 
            onPress={() => toggleTask(index)}
          >
            <Text style={item.completed ? styles.textDone : styles.textTodo}>
              {item.completed ? "✓ " : "○ "} {item.name}
            </Text>
          </TouchableOpacity>
        )}
        keyExtractor={(item, index) => index.toString()}
      />
      <TouchableOpacity style={styles.finishButton}>
        <Text style={styles.finishText}>Complete & Notify Client</Text>
      </TouchableOpacity>
    </View>
  );
};

// ... Styles for a clean, task-oriented list
