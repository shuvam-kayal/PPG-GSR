#include <Wire.h>
#include "MAX30105.h" // This library handles the MAX30102 as well

MAX30105 particleSensor;

void setup() {
  // Start the serial communication to your laptop
  Serial.begin(115200);
  Serial.println("Initializing MAX30102...");

  // Initialize the sensor (I2C)
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1); // Halt execution if sensor isn't found
  }

  byte ledBrightness = 60;   // Tuned for 5.0V VLED hardware supply
  byte sampleAverage = 8;    // Hardware averaging of 8 samples
  byte ledMode = 2;          // Red + IR mode for physiological data
  int sampleRate = 400;      // Native 400 Hz sampling rate
  int pulseWidth = 411;      // 411 µs unlocks 18-bit ADC resolution
  int adcRange = 4096;       // 4096 nA maximizes dynamic range

  // Apply base configuration
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  
}

void loop() {
  // Read the raw values
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  // Print the time-series data to the Serial Monitor
  Serial.print("IR_Value:");
  Serial.print(irValue);
  Serial.print(", Red_Value:");
  Serial.println(redValue);

  // Small delay to make the data stream manageable
  delay(50); 
}