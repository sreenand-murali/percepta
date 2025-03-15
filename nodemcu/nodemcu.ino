#include <ESP8266WiFi.h>
#include <ESPAsyncWebServer.h>

const char* ssid = "M HOSTEL5B BLOCK";
const char* password = "hostel123B5";

AsyncWebServer server(80);
unsigned long timerEnd = 0;
bool isTimerActive = false;
String pendingCommand = "";

void activatePins(String command) {
  Serial.print("Activating pins with command: ");
  Serial.println(command);
  
  int duration = command[0] - '0';  // Extract duration from first digit
  
  int gpioPins[] = {D0, D1, D2, D5, D6};

  for (int i = 1; i < command.length(); i++) {
    int pinIndex = command[i] - '1';
    if (pinIndex >= 0 && pinIndex < 5) {
      digitalWrite(gpioPins[pinIndex], HIGH);
      Serial.print("Turning ON GPIO ");
      Serial.println(gpioPins[pinIndex]);
    }
  }

  timerEnd = millis() + (duration * 1000);
  isTimerActive = true;
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  pinMode(D0, OUTPUT);
  pinMode(D1, OUTPUT);
  pinMode(D2, OUTPUT);
  pinMode(D5, OUTPUT);
  pinMode(D6, OUTPUT);
  
  digitalWrite(D0, LOW);
  digitalWrite(D1, LOW);
  digitalWrite(D2, LOW);
  digitalWrite(D5, LOW);
  digitalWrite(D6, LOW);

  server.on("/gpio", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("command")) {
      String command = request->getParam("command")->value();
      Serial.print("Received command: ");
      Serial.println(command);
      
      if (command.length() >= 2) {
        if (!isTimerActive) {
          activatePins(command);
        } else {
          pendingCommand = command;
          Serial.println("Another command is pending execution");
        }
        request->send(200, "text/plain", "Command received: " + command);
      } else {
        request->send(400, "text/plain", "Invalid command format");
        Serial.println("Error: Invalid command format");
      }
    } else {
      request->send(400, "text/plain", "Missing command parameter");
      Serial.println("Error: Missing command parameter");
    }
  });
  
  server.begin();
}

void loop() {
  if (isTimerActive && millis() >= timerEnd) {
    Serial.println("Timer ended, turning OFF all GPIOs");
    digitalWrite(D0, LOW);
    digitalWrite(D1, LOW);
    digitalWrite(D2, LOW);
    digitalWrite(D5, LOW);
    digitalWrite(D6, LOW);
    isTimerActive = false;
    
    if (pendingCommand.length() > 0) {
      Serial.print("Executing pending command: ");
      Serial.println(pendingCommand);
      activatePins(pendingCommand);
      pendingCommand = "";
    }
  }
}
