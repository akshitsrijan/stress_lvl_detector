
Arduino UNO Biofeedback Project
Components Used:

Arduino UNO

CJMCU-6701 GSR Sensor

HC-05 Bluetooth Module

Red LED

Description
This project demonstrates a biofeedback system using Arduino UNO. It reads Galvanic Skin Response (GSR) data via the CJMCU-6701 sensor and streams this data to a Bluetooth terminal app on your phone through the HC-05 Bluetooth module. It also features a red LED that blinks when the GSR value exceeds a preset threshold (default: 25), and can be controlled remotely via Bluetooth commands.

Features
Live GSR sensor readings.

Wireless Bluetooth transmission to mobile device terminal apps.

LED indicator blinks when GSR > 25.

Manual LED control via Bluetooth ('1' for ON, '0' for OFF).

Wiring
Component	Arduino Pin	Notes
CJMCU-6701 VCC	5V	Sensor power
CJMCU-6701 GND	GND	Sensor ground
CJMCU-6701 OUT	A0	Analog input for GSR sensor
HC-05 VCC	5V	Bluetooth power
HC-05 GND	GND	Bluetooth ground
HC-05 TX	Pin 2 (RX)	SoftwareSerial RX
HC-05 RX	Pin 3 (TX) via voltage divider	SoftwareSerial TX
Red LED Anode	Pin 13, via 220Ω resistor	Digital output
Red LED Cathode	GND	LED ground
Getting Started
Connect all components as per wiring above.

Upload the provided Arduino sketch (.ino file) with GSR reading and Bluetooth communication.

Install a "Bluetooth Terminal" app on your phone (e.g., Serial Bluetooth Terminal).

Pair your phone with HC-05 module (default code: 1234 or 0000).

Open the terminal app, connect to HC-05, and view live sensor data.

Example Arduino Code Snippet
cpp
#include <SoftwareSerial.h>
SoftwareSerial btSerial(2, 3); // RX, TX

int sensorPin = A0;
int ledPin = 13;

void setup() {
  Serial.begin(9600);
  btSerial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int gsrValue = analogRead(sensorPin);
  Serial.println(gsrValue);
  btSerial.println(gsrValue);

  if (gsrValue > 25) {
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(ledPin, LOW);
    delay(200);
  } else {
    digitalWrite(ledPin, LOW);
  }
  
  if(btSerial.available()) {
    char val = btSerial.read();
    if(val == '1') digitalWrite(ledPin, HIGH);
    if(val == '0') digitalWrite(ledPin, LOW);
  }

  delay(100);
}
Notes
Adjust the GSR threshold or blink timing by editing the code.

Ensure proper voltage dividing for HC-05 RX pin.

Test modules individually before assembling.

License
This project is open-source under the MIT License.
Feel free to extend or adapt!

Contact
For issues or suggestions, open a GitHub issue or reach out via email listed in the repo profile.
