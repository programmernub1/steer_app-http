# S.T.E.E.R. - Smart Teleoperated ESP32 Educational Robot

## 1. Create the project and drop in the files

```bash
flutter create steer_app --platforms=android,web
cd steer_app
# replace pubspec.yaml and lib/main.dart with the ones from this folder
flutter pub get
```

## 2. Android configuration (required for the phone camera)

Run this once from the project root (it adds the `CAMERA` + `INTERNET`
permissions, cleartext HTTP for the ESP32-CAM, and `minSdk 24`):

```bash
python3 tools/patch_android.py
```

**If the camera never starts, this is the usual cause**: without the `CAMERA`
permission in `android/app/src/main/AndroidManifest.xml` Android never shows the
permission prompt. After patching, uninstall the old app and reinstall.
iOS: add `NSCameraUsageDescription` to `ios/Runner/Info.plist`.

## 3. Run

```bash
flutter run -d chrome        # Web: runs in demo mode (UDP is impossible in browsers)
flutter run                  # Android device: real UDP, real camera
flutter build apk --release  # downloadable APK
```

Web camera requires `localhost` or HTTPS.

## 4. What is real vs mock

| Feature | Android | Web |
|---|---|---|
| UDP commands (20 Hz) + PING/PONG + Auto-Scan | real | simulated (demo robot) |
| Hand camera preview + front/rear + mirror | real | real (browser permission) |
| Live hand detection (Drive + Practice) | real (built-in camera tracker) | not available - use on-screen buttons |
| Robot ESP32-CAM view | MJPEG while an ESP32 is connected, else "No camera detected!" | always "No camera detected!" (demo) |

### Built-in hand tracker
While the Drive screen is open (normal **and** Practice mode) the app reads the
camera image stream and runs a lightweight on-device detector: skin-tone
segmentation -> largest blob -> palm centre -> radial finger count. The
21-point skeleton overlay is anchored on the detected hand. Practice mode runs
the same detection but transmits nothing to the robot.

Tips: plain background, even light, keep the hand (not your face) inside the
dashed frame, spread your fingers, keep the thumb tucked for 1-4. A fist or
5 fingers = Palm (stop). No hand for 500 ms = Hand-Lost auto-brake.
Tapping an on-screen gesture button overrides the tracker for 2 seconds.

### Plugging in MediaPipe (optional, more robust)
Call, per frame with a detected hand:

```dart
appState.onHandFrame(HandFrame(listOf21NormalizedOffsets));
```

Real landmarks take priority over the built-in tracker's overlay and use the
same debouncing and failsafe.

## 5. UDP protocol

App -> robot (ASCII, 20 Hz): `<code>,<left>,<right>`

| Gesture | code | example |
|---|---|---|
| 1 Forward | `1` | `1,140,140` |
| 2 Backward | `2` | `2,-140,-140` |
| 3 Left | `3` | `3,-140,140` |
| 4 Right | `4` | `4,140,-140` |
| Palm / E-stop | `P` | `P,0,0` |
| Hand lost | `X` | `X,0,0` |

`left/right` are signed PWM (-255..255) with speed, deadband and steering trim applied.
Heartbeat: app sends `PING` every second, robot answers `PONG,<battery%>`.

## 6. Minimal ESP32 firmware (Arduino core, L298N)

```cpp
#include <WiFi.h>
#include <WiFiUdp.h>

WiFiUDP udp;
unsigned long lastCmd = 0;
const int ENA = 25, IN1 = 26, IN2 = 27, ENB = 14, IN3 = 12, IN4 = 13;  // adjust

void setMotor(int a, int b, int en, int v) {
  digitalWrite(a, v > 0);
  digitalWrite(b, v < 0);
  analogWrite(en, abs(v));
}
void drive(int l, int r) { setMotor(IN1, IN2, ENA, l); setMotor(IN3, IN4, ENB, r); }

void setup() {
  int pins[] = {ENA, IN1, IN2, ENB, IN3, IN4};
  for (int p : pins) pinMode(p, OUTPUT);
  WiFi.softAP("STEER-ROBOT", "12345678");   // robot IP = 192.168.4.1
  udp.begin(4210);
}

void loop() {
  int n = udp.parsePacket();
  if (n) {
    char buf[32];
    int len = udp.read(buf, 31);
    buf[len] = 0;
    if (!strncmp(buf, "PING", 4)) {
      udp.beginPacket(udp.remoteIP(), udp.remotePort());
      udp.print("PONG,85");
      udp.endPacket();
    } else {
      char c; int l, r;
      if (sscanf(buf, "%c,%d,%d", &c, &l, &r) == 3) { drive(l, r); lastCmd = millis(); }
    }
  }
  if (millis() - lastCmd > 500) drive(0, 0);   // failsafe: brake if link drops
}
```

## 7. UPDATE: HTTP transport (replaces the UDP protocol above)

The app now controls the ESP32 over HTTP, matching the Arduino `WebServer` firmware:

- `GET http://<ip>:<port>/move?left=-1|0|1&right=-1|0|1&speed=0..100`
- `GET http://<ip>:<port>/status` (any HTTP reply counts as "robot alive")

The "HTTP Port" box on Connect robot must match the port your sketch serves on
(default 80). Phone and ESP32 must be on the same Wi-Fi. Auto-Scan is disabled.
The firmware must stop the motors itself if no /move arrives for ~1 second.
