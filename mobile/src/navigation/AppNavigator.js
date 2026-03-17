const AppNavigator = ({ user }) => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        {!user.token ? (
          // Auth Stack: Everyone sees this when logged out
          <Stack.Screen name="Login" component={LoginScreen} />
        ) : user.role === 'cleaner' ? (
          // Cleaner Stack: Only the crew sees these
          <Stack.Group>
            <Stack.Screen name="Schedule" component={CleanerSchedule} />
            <Stack.Screen name="JobLog" component={JobChecklistScreen} />
          </Stack.Group>
        ) : (
          // Client Stack: Only customers see these
          <Stack.Group>
            <Stack.Screen name="Services" component={ServicesScreen} />
            <Stack.Screen name="Booking" component={BookingScreen} />
            <Stack.Screen name="Payment" component={PaymentScreen} />
          </Stack.Group>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};
