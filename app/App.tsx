import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { ModelProvider } from './src/ml/ModelContext';
import CaptureScreen from './src/screens/CaptureScreen';
import ResultScreen from './src/screens/ResultScreen';
import InfoScreen from './src/screens/InfoScreen';
import LogScreen from './src/screens/LogScreen';
import ContributeScreen from './src/screens/ContributeScreen';
import AboutScreen from './src/screens/AboutScreen';
import type { ClassName } from './src/ml/contract';

export type RootStackParamList = {
  Tabs: undefined;
  Result: { imageUri: string; prediction: ClassName; confidence: number };
  Info: { className: ClassName };
};

export type TabParamList = {
  Scan: undefined;
  History: undefined;
  Contribute: undefined;
  About: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tabs = createBottomTabNavigator<TabParamList>();

function TabsNavigator() {
  return (
    <Tabs.Navigator>
      <Tabs.Screen name="Scan" component={CaptureScreen} />
      <Tabs.Screen name="History" component={LogScreen} />
      <Tabs.Screen name="Contribute" component={ContributeScreen} />
      <Tabs.Screen name="About" component={AboutScreen} />
    </Tabs.Navigator>
  );
}

export default function App() {
  return (
    <ModelProvider>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Tabs" component={TabsNavigator} options={{ headerShown: false }} />
          <Stack.Screen name="Result" component={ResultScreen} />
          <Stack.Screen name="Info" component={InfoScreen} />
        </Stack.Navigator>
      </NavigationContainer>
      <StatusBar style="light" />
    </ModelProvider>
  );
}
