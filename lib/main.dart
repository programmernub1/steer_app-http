import 'dart:async';
import 'dart:convert';
import 'dart:io'
    show
        InternetAddress,
        InternetAddressType,
        NetworkInterface,
        RawDatagramSocket,
        RawSocketEvent;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Portrait only: the hand detector assumes an upright camera image.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final state = AppState();
  await state.loadSettings();
  runApp(SteerApp(appState: state));
}

class SteerColors {
  static const Color bg = Color(0xFFF2F7F6);
  static const Color card = Colors.white;
  static const Color teal = Color(0xFF009B7D);
  static const Color tealLight = Color(0xFFE1F3EE);
  static const Color dark = Color(0xFF0F2830);
  static const Color cameraBg = Color(0xFF122832);
  static const Color subtitle = Color(0xFF536B73);
  static const Color danger = Color(0xFFD32F2F);
  static const Color dangerLight = Color(0xFFFCE8E8);
  static const Color warning = Color(0xFFF59E0B);
  static const Color warningYellow = Color(0xFFFFD54F);
}

// ============================================================================
// HAND TRACKING MODEL + GESTURE CLASSIFIER
// ============================================================================
//
// MediaPipe hand landmark order (21 points, normalized 0..1, y points down):
//  0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
//
// HOOK: when you add a real MediaPipe/TFLite plugin, call
//   appState.onHandFrame(HandFrame(landmarks))   when a hand is detected
// (and simply stop calling it when no hand is visible - the failsafe
// watchdog in AppState brakes the robot automatically after 500 ms).

class HandFrame {
  final List<Offset> landmarks; // exactly 21 normalized points
  const HandFrame(this.landmarks);
}

/// Returns '1'..'4' or 'Palm' from 21 landmarks. A closed fist maps to 'Palm'
/// (stop) because stopping is the safe default.
String classifyHand(List<Offset> lm, double sensitivity) {
  if (lm.length != 21) return 'Palm';
  double d(int a, int b) => (lm[a] - lm[b]).distance;
  // Higher sensitivity -> smaller margin needed to call a finger "extended".
  final margin = 0.30 - 0.25 * sensitivity.clamp(0.0, 1.0);
  bool ext(int tip, int pip) => d(0, tip) > d(0, pip) * (1 + margin);

  final count = [ext(8, 6), ext(12, 10), ext(16, 14), ext(20, 18)]
      .where((e) => e)
      .length;
  final thumbOut = d(4, 17) > d(3, 17) * (1 + margin * 0.5);

  if (count == 0) return 'Palm';
  if (count == 4 && thumbOut) return 'Palm';
  return '$count';
}

const List<List<int>> kHandConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/// Result of the on-device hand detector for one camera frame.
class HandObservation {
  final String gesture; // '1'..'4' or 'Palm'
  final int fingers; // raw finger count (0..5)
  final Offset center; // palm centre, normalized to the upright image (0..1)
  final double radiusNormH; // palm radius as a fraction of image height
  final double aspect; // upright image width / height
  const HandObservation({
    required this.gesture,
    required this.fingers,
    required this.center,
    required this.radiusNormH,
    required this.aspect,
  });
}

int _clampInt(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

/// Lightweight, dependency-free hand detector that runs on the live camera
/// image stream: skin-tone segmentation (YCbCr) -> largest blob -> palm centre
/// (distance transform) -> radial scan to count raised fingers.
///
/// It is a practical stand-in for MediaPipe: it works best with a plain
/// background, good light, spread fingers, thumb tucked for 1-4, and the face
/// outside the guide frame. For production-grade tracking, feed real landmarks
/// through AppState.onHandFrame() instead.
class SkinHandDetector {
  static const int gw = 72; // analysis grid width (cells)
  int _gh = 0;
  Uint8List _mask = Uint8List(0);
  Int32List _queue = Int32List(0);
  Int32List _dt = Int32List(0);

  int _flood(int seed, int from, int to) {
    var head = 0, tail = 0;
    _queue[tail++] = seed;
    _mask[seed] = to;
    while (head < tail) {
      final c = _queue[head++];
      final x = c % gw;
      final y = c ~/ gw;
      if (x > 0 && _mask[c - 1] == from) {
        _mask[c - 1] = to;
        _queue[tail++] = c - 1;
      }
      if (x < gw - 1 && _mask[c + 1] == from) {
        _mask[c + 1] = to;
        _queue[tail++] = c + 1;
      }
      if (y > 0 && _mask[c - gw] == from) {
        _mask[c - gw] = to;
        _queue[tail++] = c - gw;
      }
      if (y < _gh - 1 && _mask[c + gw] == from) {
        _mask[c + gw] = to;
        _queue[tail++] = c + gw;
      }
    }
    return tail;
  }

  HandObservation? detect(
      CameraImage img, int sensorRotation, double sensitivity) {
    final sw = img.width, sh = img.height;
    final rot = ((sensorRotation % 360) + 360) % 360;
    final swap = rot == 90 || rot == 270;
    final uw = swap ? sh : sw; // upright (portrait) image size
    final uh = swap ? sw : sh;
    final gh = (gw * uh / uw).round();
    final n = gw * gh;
    if (_gh != gh) {
      _gh = gh;
      _mask = Uint8List(n);
      _queue = Int32List(n);
      _dt = Int32List(n);
    } else {
      _mask.fillRange(0, n, 0);
    }

    final planes = img.planes;
    final bgra = planes.length == 1;
    if (!bgra && planes.length < 3) return null;
    final p0 = planes[0].bytes;
    final r0 = planes[0].bytesPerRow;
    final pu = bgra ? p0 : planes[1].bytes;
    final pv = bgra ? p0 : planes[2].bytes;
    final ru = bgra ? 0 : planes[1].bytesPerRow;
    final rv = bgra ? 0 : planes[2].bytesPerRow;
    final su = bgra ? 1 : (planes[1].bytesPerPixel ?? 1);
    final sv = bgra ? 1 : (planes[2].bytesPerPixel ?? 1);

    // Region of interest (keeps the face / background out as much as possible).
    final x0 = (gw * 0.12).round(), x1 = (gw * 0.88).round();
    final y0 = (gh * 0.20).round(), y1 = (gh * 0.95).round();
    final roiCells = (x1 - x0) * (y1 - y0);
    var skin = 0;

    for (var gy = y0; gy < y1; gy++) {
      final uy = _clampInt(((gy + 0.5) * uh / gh).floor(), 0, uh - 1);
      for (var gx = x0; gx < x1; gx++) {
        final ux = _clampInt(((gx + 0.5) * uw / gw).floor(), 0, uw - 1);
        int sx, sy;
        switch (rot) {
          case 90:
            sx = uy;
            sy = sh - 1 - ux;
            break;
          case 180:
            sx = sw - 1 - ux;
            sy = sh - 1 - uy;
            break;
          case 270:
            sx = sw - 1 - uy;
            sy = ux;
            break;
          default:
            sx = ux;
            sy = uy;
        }
        int yv, cb, cr;
        if (bgra) {
          final i = sy * r0 + sx * 4;
          if (i + 2 >= p0.length) continue;
          final b = p0[i], g = p0[i + 1], r = p0[i + 2];
          yv = (0.299 * r + 0.587 * g + 0.114 * b).round();
          cb = (128 - 0.168736 * r - 0.331264 * g + 0.5 * b).round();
          cr = (128 + 0.5 * r - 0.418688 * g - 0.081312 * b).round();
        } else {
          final iy = sy * r0 + sx;
          final iu = (sy >> 1) * ru + (sx >> 1) * su;
          final iv = (sy >> 1) * rv + (sx >> 1) * sv;
          if (iy >= p0.length || iu >= pu.length || iv >= pv.length) continue;
          yv = p0[iy];
          cb = pu[iu];
          cr = pv[iv];
        }
        if (yv > 40 && cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173) {
          _mask[gy * gw + gx] = 1;
          skin++;
        }
      }
    }
    if (skin < roiCells * 0.03) return null;

    // Keep only the largest skin blob.
    var bestSeed = -1, bestArea = 0;
    for (var i = 0; i < n; i++) {
      if (_mask[i] != 1) continue;
      final area = _flood(i, 1, 2);
      if (area > bestArea) {
        bestArea = area;
        bestSeed = i;
      }
    }
    if (bestSeed < 0 || bestArea < roiCells * 0.03 || bestArea > roiCells * 0.65) {
      return null; // nothing hand-sized (or something far too big: too close)
    }
    for (var i = 0; i < n; i++) {
      if (_mask[i] == 2) _mask[i] = 1;
    }
    _flood(bestSeed, 1, 3);
    for (var i = 0; i < n; i++) {
      if (_mask[i] == 1) _mask[i] = 0;
    }

    // Chamfer (3-4) distance transform -> palm centre = deepest point.
    const big = 1 << 28;
    for (var y = 0; y < gh; y++) {
      for (var x = 0; x < gw; x++) {
        final i = y * gw + x;
        if (_mask[i] != 3) {
          _dt[i] = 0;
          continue;
        }
        var d = big;
        d = math.min(d, (x > 0 ? _dt[i - 1] : 0) + 3);
        d = math.min(d, (y > 0 ? _dt[i - gw] : 0) + 3);
        d = math.min(d, (x > 0 && y > 0 ? _dt[i - gw - 1] : 0) + 4);
        d = math.min(d, (x < gw - 1 && y > 0 ? _dt[i - gw + 1] : 0) + 4);
        _dt[i] = d;
      }
    }
    var maxD = 0, bestIdx = bestSeed;
    for (var y = gh - 1; y >= 0; y--) {
      for (var x = gw - 1; x >= 0; x--) {
        final i = y * gw + x;
        if (_mask[i] != 3) continue;
        var d = _dt[i];
        d = math.min(d, (x < gw - 1 ? _dt[i + 1] : 0) + 3);
        d = math.min(d, (y < gh - 1 ? _dt[i + gw] : 0) + 3);
        d = math.min(d, (x < gw - 1 && y < gh - 1 ? _dt[i + gw + 1] : 0) + 4);
        d = math.min(d, (x > 0 && y < gh - 1 ? _dt[i + gw - 1] : 0) + 4);
        _dt[i] = d;
        if (d > maxD) {
          maxD = d;
          bestIdx = i;
        }
      }
    }
    final rin = maxD / 3.0;
    if (rin < 3.0) return null;
    final cx = (bestIdx % gw) + 0.5;
    final cy = (bestIdx ~/ gw) + 0.5;

    // Radial scan over the upper arc (+-110 deg from "up"): count protrusions.
    final thr = (1.75 - 0.5 * math.max(0.0, math.min(1.0, sensitivity))) * rin;
    final maxT = math.max(gw, gh).toDouble();
    final samples = <double>[];
    for (var deg = -110; deg <= 110; deg += 3) {
      final a = deg * math.pi / 180.0;
      final dx = math.sin(a), dy = -math.cos(a);
      var last = 0.0;
      var miss = 0;
      for (var t = 0.0; t < maxT; t += 0.5) {
        final ix = (cx + dx * t).floor();
        final iy = (cy + dy * t).floor();
        final inside =
            ix >= 0 && ix < gw && iy >= 0 && iy < gh && _mask[iy * gw + ix] == 3;
        if (inside) {
          last = t;
          miss = 0;
        } else {
          miss++;
          if (miss > 2) break;
        }
      }
      samples.add(last);
    }
    var fingers = 0, run = 0;
    for (var k = 0; k <= samples.length; k++) {
      final above = k < samples.length && samples[k] > thr;
      if (above) {
        run++;
      } else {
        if (run >= 2) fingers += math.max(1, (run / 5.0).round());
        run = 0;
      }
    }

    final gesture = (fingers == 0 || fingers >= 5) ? 'Palm' : '$fingers';
    return HandObservation(
      gesture: gesture,
      fingers: fingers,
      center: Offset(cx / gw, cy / gh),
      radiusNormH: rin / gh,
      aspect: uw / uh,
    );
  }
}

/// Which fingers are raised for a gesture (0 thumb, 1 index ... 4 pinky).
Set<int> extendedFingers(String g) {
  switch (g) {
    case '1':
      return {1};
    case '2':
      return {1, 2};
    case '3':
      return {1, 2, 3};
    case '4':
      return {1, 2, 3, 4};
    case 'Palm':
      return {0, 1, 2, 3, 4};
    default:
      return <int>{};
  }
}

/// 21-point hand skeleton (MediaPipe index order) in palm-radius units,
/// centred on the palm, y pointing down. Raised fingers are extended.
List<Offset> buildHandTemplate(String gesture) {
  final ext = extendedFingers(gesture);
  final pts = <Offset>[const Offset(0, 1.5)]; // 0 wrist
  if (ext.contains(0)) {
    pts.addAll(const [
      Offset(-0.7, 1.0),
      Offset(-1.15, 0.55),
      Offset(-1.55, 0.1),
      Offset(-1.95, -0.3),
    ]);
  } else {
    pts.addAll(const [
      Offset(-0.7, 1.0),
      Offset(-1.0, 0.6),
      Offset(-0.8, 0.2),
      Offset(-0.45, 0.05),
    ]);
  }
  const baseX = [-0.65, -0.2, 0.25, 0.7];
  const baseY = [-0.5, -0.6, -0.5, -0.3];
  const splay = [-0.12, -0.03, 0.06, 0.15];
  const seg = [
    [0.75, 0.55, 0.5],
    [0.85, 0.6, 0.5],
    [0.75, 0.55, 0.5],
    [0.55, 0.4, 0.4],
  ];
  for (var i = 0; i < 4; i++) {
    final base = Offset(baseX[i], baseY[i]);
    pts.add(base);
    if (ext.contains(i + 1)) {
      final raw = Offset(splay[i], -1);
      final dir = raw / raw.distance;
      var cur = base;
      for (var j = 0; j < 3; j++) {
        cur = cur + dir * seg[i][j];
        pts.add(cur);
      }
    } else {
      final pip = base + const Offset(0, -0.35);
      final dip = pip + const Offset(0, 0.25);
      final tip = dip + const Offset(0, 0.2);
      pts.addAll([pip, dip, tip]);
    }
  }
  return pts;
}

// ============================================================================
// APP STATE: network (HTTP), camera, MJPEG, gestures, calibration
// ============================================================================
//
// UDP PACKET FORMAT (ASCII, sent at 20 Hz while connected):
//     "<code>,<left>,<right>"      e.g.  "1,140,140"   "3,-140,140"   "P,0,0"
//   code  : '1' fwd | '2' back | '3' left | '4' right | 'P' palm/stop | 'X' hand lost
//   left/right : signed PWM -255..255 (deadband, speed and trim already applied)
// Heartbeat: app sends "PING" every second; robot should answer "PONG[,battery%]".
// The robot should brake if no packet arrives for ~500 ms.

/// Lifecycle of the phone camera, used to show a spinner / permission prompt.
enum CameraStatus { idle, initializing, ready, denied, error }

class AppState extends ChangeNotifier {
  // Network & Hardware State
  String robotIp = '192.168.4.1';
  String udpPort = '80'; // HTTP control port
  String camStreamUrl = 'http://192.168.4.2:81/stream';
  bool isConnected = false;
  bool hasTestedConnection = false;
  bool isScanning = false;
  bool isTesting = false;
  String? statusMessage;
  int linkQuality = 0;
  int batteryLevel = 78;
  int latencyMs = 0;

  /// Demo mode: simulate the robot instead of sending real UDP. Always on for Web.
  bool mockMode = kIsWeb;

  /// Treat the robot as connected even if it never answers PING (one-way UDP).
  bool oneWayMode = false;

  bool get isMock => mockMode || kIsWeb;

  // Camera & Vision State
  bool isFrontCamera = true;
  bool mirrorCamera = true;
  bool showLandmarks = true;
  bool isPipSwapped = false;
  bool isHorizontalSplit = false;
  bool isPracticeMode = false;
  List<Offset>? handLandmarks; // real landmarks, when a tracker feeds us

  // Active Command ('1', '2', '3', '4', 'Palm', 'LOST')
  String activeGesture = 'Palm';
  bool estop = false;

  // Sensitivity & Motor Calibration
  double gestureSensitivity = 0.70;
  double maxMotorSpeed = 0.55;
  double steeringTrim = 0.0; // -10 to +10
  double minPwmDeadband = 0.30; // 0.15 to 0.50

  final ValueNotifier<String> packetNotifier = ValueNotifier('P,0,0');
  final ValueNotifier<Uint8List?> robotFrame = ValueNotifier(null);
  bool streamLive = false;

  // --- internals ---
  final math.Random _rng = math.Random();
  RawDatagramSocket? _socket; // unused now (kept so old references compile)
  final http.Client _ctl = http.Client(); // HTTP control client
  String _lastSent = '';
  DateTime _lastSentAt = DateTime.fromMillisecondsSinceEpoch(0);
  InternetAddress? _addr;
  int _port = 80;
  Completer<void>? _pongWaiter;
  DateTime? _lastPingAt;
  DateTime? _lastPong;
  int _mockTicks = 0;
  Timer? _txTimer;
  Timer? _beatTimer;
  Timer? _watchTimer;

  // tracker debounce / failsafe
  String? _candidate;
  int _candidateCount = 0;
  DateTime _lastHandSeen = DateTime.now();
  bool _trackerLive = false;

  // camera
  CameraController? camController;
  List<CameraDescription> _cams = [];
  String? cameraError;
  CameraStatus cameraStatus = CameraStatus.idle;
  bool _camReinitPending = false;

  // on-device hand tracking (camera image stream -> SkinHandDetector)
  final SkinHandDetector _detector = SkinHandDetector();
  HandObservation? handObs;
  bool trackingActive = false;
  int _trackRefs = 0;
  bool _syncBusy = false;
  bool _syncAgain = false;
  DateTime _lastProc = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _manualHoldUntil = DateTime.fromMillisecondsSinceEpoch(0);
  bool get trackerSupported => !kIsWeb;

  /// True when the robot camera stream was expected but could not be read.
  bool streamError = false;
  int _camRefs = 0;
  bool _camBusy = false;

  // mjpeg
  int _streamRefs = 0;
  int _streamGen = 0;
  http.Client? _mjpegClient;

  AppState() {
    _txTimer = Timer.periodic(const Duration(milliseconds: 50), (_) => _tick());
    _beatTimer = Timer.periodic(const Duration(seconds: 1), (_) => _beat());
    _watchTimer =
        Timer.periodic(const Duration(milliseconds: 100), (_) => _watch());
  }

  void _notifyLater() => scheduleMicrotask(notifyListeners);

  // ------------------------------------------------------------ persistence
  Future<void> loadSettings() async {
    try {
      final p = await SharedPreferences.getInstance();
      robotIp = p.getString('ip') ?? robotIp;
      udpPort = p.getString('port') ?? udpPort;
      camStreamUrl = p.getString('cam') ?? camStreamUrl;
      gestureSensitivity = p.getDouble('sens') ?? gestureSensitivity;
      maxMotorSpeed = p.getDouble('speed') ?? maxMotorSpeed;
      steeringTrim = p.getDouble('trim') ?? steeringTrim;
      minPwmDeadband = p.getDouble('dead') ?? minPwmDeadband;
      mockMode = kIsWeb || (p.getBool('mock') ?? false);
      oneWayMode = p.getBool('oneway') ?? false;
    } catch (_) {}
  }

  Future<void> saveSettings() async {
    try {
      final p = await SharedPreferences.getInstance();
      await p.setString('ip', robotIp);
      await p.setString('port', udpPort);
      await p.setString('cam', camStreamUrl);
      await p.setDouble('sens', gestureSensitivity);
      await p.setDouble('speed', maxMotorSpeed);
      await p.setDouble('trim', steeringTrim);
      await p.setDouble('dead', minPwmDeadband);
      await p.setBool('mock', mockMode);
      await p.setBool('oneway', oneWayMode);
    } catch (_) {}
  }

  // ------------------------------------------------------------ commands
  static String nameFor(String g) {
    switch (g) {
      case '1':
        return 'Forward';
      case '2':
        return 'Backward';
      case '3':
        return 'Turn left';
      case '4':
        return 'Turn right';
      case 'Palm':
        return 'Stop';
      case 'LOST':
        return 'Auto-Brake (No Hand)';
      default:
        return 'Stop';
    }
  }

  String get commandName => estop ? 'Stop' : nameFor(activeGesture);

  /// Live hand-tracking feedback line (shown in Drive and Practice mode).
  String get trackingLabel {
    if (!trackerSupported) {
      return 'Hand tracking runs in the Android app (use buttons in browser)';
    }
    if (!trackingActive) {
      return cameraStatus == CameraStatus.ready
          ? 'Starting hand tracking...'
          : 'Waiting for camera...';
    }
    final o = handObs;
    if (o == null) return 'No hand detected';
    final f = o.fingers > 5 ? 5 : o.fingers;
    return 'Hand: $f finger${f == 1 ? "" : "s"} -> ${nameFor(o.gesture)}';
  }

  bool get isMoving =>
      !isPracticeMode &&
      !estop &&
      (activeGesture == '1' ||
          activeGesture == '2' ||
          activeGesture == '3' ||
          activeGesture == '4');

  /// Manual gesture selection (on-screen buttons). Also clears an e-stop latch.
  void setGesture(String gesture) {
    estop = false;
    activeGesture = gesture;
    // Let a manual tap win over the camera tracker for a couple of seconds.
    _manualHoldUntil = DateTime.now().add(const Duration(seconds: 2));
    notifyListeners();
  }

  void emergencyStop() {
    estop = true;
    activeGesture = 'Palm';
    _burstStop();
    notifyListeners();
  }

  /// Silent stop (no notify) - safe to call from dispose().
  void sendStopBurst() {
    activeGesture = 'Palm';
    _burstStop();
  }

  void onAppPaused() {
    if (isConnected) sendStopBurst();
  }

  void _burstStop() {
    packetNotifier.value = 'P,0,0';
    if (isConnected && !isMock) {
      for (var i = 0; i < 3; i++) {
        _sendRaw('P,0,0');
      }
    }
  }

  String _codeFor(String g) => g == 'Palm' ? 'P' : (g == 'LOST' ? 'X' : g);

  /// Applies speed cap, minimum-PWM deadband and steering trim.
  (int, int) _mix(String g) {
    final mag = math.max(maxMotorSpeed, minPwmDeadband);
    final pwm = mag * 255.0;
    final t = (steeringTrim / 10.0) * 0.15; // up to 15% differential
    double l = 0, r = 0;
    switch (g) {
      case '1':
        l = pwm * (1 + t);
        r = pwm * (1 - t);
        break;
      case '2':
        l = -pwm * (1 + t);
        r = -pwm * (1 - t);
        break;
      case '3':
        l = -pwm;
        r = pwm;
        break;
      case '4':
        l = pwm;
        r = -pwm;
        break;
    }
    int c(double v) => v.round().clamp(-255, 255);
    return (c(l), c(r));
  }

  String get currentPacket {
    final g = (estop || isPracticeMode) ? 'Palm' : activeGesture;
    final m = _mix(g);
    return '${_codeFor(g)},${m.$1},${m.$2}';
  }

  // ------------------------------------------------------------ tracker feed
  /// Optional: feed real MediaPipe landmarks from a plugin (see HandFrame notes).
  void onHandFrame(HandFrame? frame) {
    if (frame == null) return; // watchdog handles absence
    _lastHandSeen = DateTime.now();
    _trackerLive = true;
    handLandmarks = frame.landmarks;
    _applyObserved(classifyHand(frame.landmarks, gestureSensitivity));
    notifyListeners();
  }

  /// Called by the built-in camera tracker for every frame with a hand.
  void onHandObservation(HandObservation o) {
    _lastHandSeen = DateTime.now();
    _trackerLive = true;
    handObs = o;
    handLandmarks = null;
    _applyObserved(o.gesture);
    notifyListeners();
  }

  void _applyObserved(String g) {
    if (DateTime.now().isBefore(_manualHoldUntil)) return;
    final need = (6 - 5 * gestureSensitivity).round().clamp(1, 6);
    if (g == _candidate) {
      _candidateCount++;
    } else {
      _candidate = g;
      _candidateCount = 1;
    }
    if (_candidateCount >= need && g != activeGesture) {
      if (estop && g != 'Palm') return;
      if (g == 'Palm') estop = false;
      activeGesture = g;
    }
  }

  void resetTracker() {
    _trackerLive = false;
    handLandmarks = null;
    handObs = null;
    _candidate = null;
    _candidateCount = 0;
  }

  void _watch() {
    // Hand-lost failsafe: tracker was live but silent for >500 ms.
    if (_trackerLive &&
        activeGesture != 'LOST' &&
        !DateTime.now().isBefore(_manualHoldUntil) &&
        DateTime.now().difference(_lastHandSeen) >
            const Duration(milliseconds: 500)) {
      activeGesture = 'LOST';
      handLandmarks = null;
      notifyListeners();
    }
  }

  void _onCameraImage(CameraImage img) {
    final now = DateTime.now();
    if (now.difference(_lastProc) < const Duration(milliseconds: 70)) return;
    _lastProc = now;
    try {
      final rot = camController?.description.sensorOrientation ?? 90;
      final obs = _detector.detect(img, rot, gestureSensitivity);
      if (obs != null) {
        onHandObservation(obs);
      } else if (handObs != null) {
        handObs = null;
        notifyListeners();
      }
    } catch (_) {}
  }

  // ------------------------------------------------------------ UDP transport
  void _tick() {
    if (!isConnected) return;
    final p = currentPacket;
    packetNotifier.value = p;
    if (isPracticeMode) return; // practice mode: nothing is transmitted
    if (!isMock) _sendRaw(p);
  }

  void _beat() {
    if (isMock) {
      if (isConnected) {
        latencyMs = 40 + _rng.nextInt(50);
        linkQuality = 88 + _rng.nextInt(9);
        _mockTicks++;
        if (_mockTicks % 30 == 0 && batteryLevel > 5) batteryLevel--;
        notifyListeners();
      }
      return;
    }
    if (_addr == null) return;
    if (isConnected || oneWayMode) {
      _lastPingAt = DateTime.now();
      _sendRaw('PING');
    }
    if (isConnected &&
        !oneWayMode &&
        _lastPong != null &&
        DateTime.now().difference(_lastPong!) > const Duration(seconds: 4)) {
      isConnected = false;
      linkQuality = 0;
      statusMessage = 'Link lost - no reply from robot';
      notifyListeners();
    }
  }

  /// HTTP endpoint the ESP32 firmware must serve.
  /// Request: GET http://<ip>:<port>/move?left=-1|0|1&right=-1|0|1&speed=0..100
  /// Health:  GET http://<ip>:<port>/status   (any HTTP reply counts as alive)
  static const String kMovePath = '/move';
  static const String kStatusPath = '/status';

  void _sendRaw(String s) {
    if (_addr == null) return;
    if (s == 'PING') {
      unawaited(_httpPing());
      return;
    }
    final parts = s.split(',');
    if (parts.length != 3) return;
    final l = int.tryParse(parts[1]) ?? 0;
    final r = int.tryParse(parts[2]) ?? 0;

    // Send on change, otherwise repeat every 300 ms as a heartbeat so the
    // robot's own timeout never trips while we are still driving.
    final key = '$l,$r';
    final now = DateTime.now();
    if (key == _lastSent &&
        now.difference(_lastSentAt) < const Duration(milliseconds: 300)) {
      return;
    }
    _lastSent = key;
    _lastSentAt = now;

    final speed = ((math.max(l.abs(), r.abs()) / 255.0) * 100).round();
    final uri = Uri.parse('http://$robotIp:$_port$kMovePath').replace(
      queryParameters: {
        'left': '${l.sign}',
        'right': '${r.sign}',
        'speed': '$speed',
      },
    );
    unawaited(_ctl
        .get(uri)
        .timeout(const Duration(milliseconds: 800))
        .then<void>((_) {})
        .catchError((_) {}));
  }

  Future<void> _httpPing() async {
    try {
      final sw = Stopwatch()..start();
      await _ctl
          .get(Uri.parse('http://$robotIp:$_port$kStatusPath'))
          .timeout(const Duration(milliseconds: 1500));
      latencyMs = sw.elapsedMilliseconds;
      linkQuality = (100 - latencyMs ~/ 4).clamp(5, 100);
      _lastPong = DateTime.now();
      notifyListeners();
    } catch (_) {}
  }

  /// Resolves the target address/port (no socket is needed for HTTP).
  Future<bool> _openSocket() async {
    if (kIsWeb) return false;
    final addr = InternetAddress.tryParse(robotIp);
    final port = int.tryParse(udpPort);
    if (addr == null || port == null || port < 1 || port > 65535) {
      statusMessage = 'Invalid IP address or port';
      return false;
    }
    _addr = addr;
    _port = port;
    return true;
  }

  Future<bool> _handshake() async {
    if (!await _openSocket()) return false;
    try {
      final sw = Stopwatch()..start();
      // Any HTTP reply, even a 404, proves the ESP32 is reachable.
      await _ctl
          .get(Uri.parse('http://$robotIp:$_port$kStatusPath'))
          .timeout(const Duration(seconds: 2));
      latencyMs = sw.elapsedMilliseconds;
      linkQuality = (100 - latencyMs ~/ 4).clamp(5, 100);
      _lastPong = DateTime.now();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<void> testConnection(String ip, String port, String camUrl) async {
    robotIp = ip.trim();
    udpPort = port.trim().isEmpty ? '80' : port.trim();
    camStreamUrl = camUrl.trim();
    hasTestedConnection = true;
    isTesting = true;
    statusMessage = null;
    notifyListeners();
    restartStream();

    if (isMock) {
      await Future.delayed(const Duration(milliseconds: 450));
      isConnected = robotIp.isNotEmpty;
      latencyMs = 40 + _rng.nextInt(50);
      linkQuality = 90;
      if (!isConnected) statusMessage = 'Camera-only demo (no robot address)';
    } else if (robotIp.isEmpty) {
      isConnected = false;
      statusMessage = 'Camera-only demo (no robot address)';
    } else {
      final ok = await _handshake();
      isConnected = ok || (oneWayMode && _addr != null);
      if (!ok) {
        statusMessage = oneWayMode
            ? 'No reply - sending one-way'
            : 'No reply from robot. Check IP, port and that phone and ESP32 are on the same Wi-Fi';
      }
    }
    isTesting = false;
    saveSettings();
    notifyListeners();
  }

  Future<void> scanNetwork() async {
    isScanning = true;
    statusMessage = null;
    notifyListeners();
    if (isMock) {
      await Future.delayed(const Duration(milliseconds: 900));
      robotIp = '192.168.4.1';
      camStreamUrl = 'http://192.168.4.2:81/stream';
      isConnected = true;
      hasTestedConnection = true;
    } else {
      final found = await _scanLan();
      hasTestedConnection = true;
      if (found != null) {
        robotIp = found;
        isConnected = await _handshake();
      } else {
        statusMessage ??= 'No robot found. Type the ESP32 IP.';
      }
    }
    isScanning = false;
    saveSettings();
    notifyListeners();
  }

  Future<String?> _scanLan() async {
    // HTTP probing would also hit routers and printers, so auto-scan is off.
    // Type the IP printed in the Arduino Serial Monitor instead.
    statusMessage = 'Auto-Scan is not available over HTTP. Type the ESP32 IP.';
    return null;
  }

  void setMockMode(bool v) {
    mockMode = kIsWeb || v;
    isConnected = false;
    statusMessage = null;
    saveSettings();
    notifyListeners();
  }

  void setOneWayMode(bool v) {
    oneWayMode = v;
    saveSettings();
    notifyListeners();
  }

  // ------------------------------------------------------------ camera
  Future<void> acquireCamera() async {
    _camRefs++;
    if (camController != null || _camBusy) return;
    await _initCamera();
  }

  void releaseCamera() {
    _camRefs = math.max(0, _camRefs - 1);
    if (_camRefs == 0) {
      final c = camController;
      camController = null;
      cameraStatus = CameraStatus.idle;
      _notifyLater();
      Future.delayed(const Duration(milliseconds: 150), () => _disposeController(c));
    }
  }

  /// Initializes the physical camera for the currently selected lens.
  /// Requests camera permission (the plugin shows the system / browser prompt)
  /// and records a status the UI uses for the spinner or permission message.
  Future<void> _initCamera() async {
    _camBusy = true;
    cameraStatus = CameraStatus.initializing;
    cameraError = null;
    _notifyLater(); // may run during initState, so never notify synchronously
    try {
      if (_cams.isEmpty) _cams = await availableCameras();
      if (_cams.isEmpty) throw StateError('no cameras');
      final want =
          isFrontCamera ? CameraLensDirection.front : CameraLensDirection.back;
      final desc = _cams.firstWhere(
        (c) => c.lensDirection == want,
        orElse: () => _cams.first,
      );
      // Keep the UI honest if the device lacks the requested lens.
      isFrontCamera = desc.lensDirection == CameraLensDirection.front;
      final fmt = defaultTargetPlatform == TargetPlatform.iOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420;
      CameraController? ready;
      Object? lastErr;
      // Attempt 1: medium + YUV (needed for hand tracking).
      // Attempt 2: low + plugin defaults, for devices that reject attempt 1.
      for (var attempt = 0; attempt < 2 && ready == null; attempt++) {
        final cc = CameraController(
          desc,
          attempt == 0 ? ResolutionPreset.medium : ResolutionPreset.low,
          enableAudio: false,
          imageFormatGroup: attempt == 0 ? fmt : null,
        );
        try {
          await cc.initialize(); // triggers the permission prompt on first use
          ready = cc;
        } on CameraException catch (e) {
          lastErr = e;
          try {
            await cc.dispose();
          } catch (_) {}
          if (_isDenied(e)) break;
        }
      }
      final c = ready;
      if (c == null) throw lastErr ?? StateError('camera init failed');
      if (_camRefs == 0) {
        await c.dispose();
        cameraStatus = CameraStatus.idle;
        return;
      }
      camController = c;
      cameraStatus = CameraStatus.ready;
    } on CameraException catch (e) {
      camController = null;
      final denied = _isDenied(e);
      cameraStatus = denied ? CameraStatus.denied : CameraStatus.error;
      cameraError = denied
          ? (kIsWeb
              ? 'Camera blocked (${e.code}). Allow it in the browser address bar; browsers need https or localhost.'
              : 'Camera permission denied (${e.code}). If no prompt appeared, the CAMERA permission is missing from AndroidManifest.xml (run tools/patch_android.py and reinstall).')
          : '${e.code}: ${e.description ?? "camera error"}';
    } catch (e) {
      camController = null;
      cameraStatus = CameraStatus.error;
      cameraError = 'Camera error: $e';
    } finally {
      _camBusy = false;
      notifyListeners();
      unawaited(_syncTracking());
      if (_camReinitPending && _camRefs > 0) {
        _camReinitPending = false;
        final old = camController;
        camController = null;
        await _disposeController(old);
        await _initCamera();
      }
    }
  }

  static bool _isDenied(CameraException e) =>
      e.code.startsWith('CameraAccessDenied') ||
      e.code == 'CameraAccessRestricted' ||
      e.code == 'cameraPermission';

  /// Switches between the physical front and rear lenses.
  Future<void> toggleFrontCamera() async {
    isFrontCamera = !isFrontCamera;
    notifyListeners();
    if (_camRefs == 0) return;
    if (_camBusy) {
      _camReinitPending = true; // re-run with the new lens when init finishes
      return;
    }
    final old = camController;
    camController = null;
    cameraStatus = CameraStatus.initializing;
    notifyListeners();
    await _disposeController(old);
    await _initCamera();
  }

  /// "Retry" button on the permission / error panel.
  Future<void> retryCamera() async {
    if (_camBusy || _camRefs == 0) return;
    final old = camController;
    camController = null;
    await _disposeController(old);
    await _initCamera();
  }

  /// Release the camera while the app is in the background (plugin guidance).
  void onAppBackgrounded() {
    final c = camController;
    if (c == null) return;
    camController = null;
    cameraStatus = CameraStatus.idle;
    trackingActive = false;
    handObs = null;
    notifyListeners();
    Future.delayed(const Duration(milliseconds: 100), () => _disposeController(c));
  }

  Future<void> _disposeController(CameraController? c) async {
    if (c == null) return;
    trackingActive = false;
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
    } catch (_) {}
    try {
      await c.dispose();
    } catch (_) {}
  }

  // --- hand tracking on the camera image stream (Drive + Practice) ---
  void acquireTracking() {
    _trackRefs++;
    unawaited(_syncTracking());
  }

  void releaseTracking() {
    _trackRefs = math.max(0, _trackRefs - 1);
    if (_trackRefs == 0) resetTracker();
    unawaited(_syncTracking());
  }

  Future<void> _syncTracking() async {
    if (_syncBusy) {
      _syncAgain = true;
      return;
    }
    _syncBusy = true;
    try {
      do {
        _syncAgain = false;
        await _syncOnce();
      } while (_syncAgain);
    } finally {
      _syncBusy = false;
    }
  }

  Future<void> _syncOnce() async {
    final c = camController;
    if (c == null) {
      trackingActive = false;
      return;
    }
    final want = _trackRefs > 0 && trackerSupported && c.value.isInitialized;
    final streaming = c.value.isStreamingImages;
    if (want && !streaming) {
      try {
        await c.startImageStream(_onCameraImage);
        trackingActive = true;
        _trackerLive = true; // from now on, no hand for 500 ms => auto-brake
        _lastHandSeen = DateTime.now();
      } catch (_) {
        trackingActive = false;
      }
      _notifyLater();
    } else if (!want && streaming) {
      try {
        await c.stopImageStream();
      } catch (_) {}
      trackingActive = false;
      _notifyLater();
    }
  }

  /// Re-open the camera on resume (also picks up permission granted in Settings).
  void onAppResumed() {
    if (_camRefs > 0 && camController == null && !_camBusy) _initCamera();
  }

  // ------------------------------------------------------------ MJPEG stream
  void acquireStream() {
    _streamRefs++;
    if (_streamRefs == 1) _runStream();
  }

  void releaseStream() {
    _streamRefs = math.max(0, _streamRefs - 1);
    if (_streamRefs == 0) _stopStream();
  }

  void restartStream() {
    if (_streamRefs > 0) {
      _stopStream();
      _runStream();
    }
  }

  void _stopStream() {
    _streamGen++;
    _mjpegClient?.close();
    _mjpegClient = null;
    streamLive = false;
    streamError = false;
    scheduleMicrotask(() {
      robotFrame.value = null;
      notifyListeners();
    });
  }

  static int _find(List<int> b, int a, int c, int from) {
    for (var i = from; i < b.length - 1; i++) {
      if (b[i] == a && b[i + 1] == c) return i;
    }
    return -1;
  }

  /// Reads the ESP32-CAM MJPEG stream, but only while a real ESP32 robot is
  /// connected. While offline / in demo mode no request is made.
  Future<void> _runStream() async {
    final gen = ++_streamGen;
    while (gen == _streamGen) {
      final robotOnline = isConnected && !isMock;
      final uri = Uri.tryParse(camStreamUrl);
      if (!robotOnline ||
          camStreamUrl.isEmpty ||
          uri == null ||
          !uri.hasScheme) {
        final err = robotOnline; // online, but no usable stream address
        if (robotFrame.value != null || streamLive || streamError != err) {
          robotFrame.value = null;
          streamLive = false;
          streamError = err;
          notifyListeners();
        }
        await Future.delayed(const Duration(milliseconds: 700));
        continue;
      }

      final client = http.Client();
      _mjpegClient = client;
      final buf = <int>[];
      try {
        final resp = await client
            .send(http.Request('GET', uri))
            .timeout(const Duration(seconds: 5));
        if (resp.statusCode != 200) throw StateError('HTTP ${resp.statusCode}');
        await for (final chunk in resp.stream) {
          if (gen != _streamGen || !isConnected) break;
          buf.addAll(chunk);
          while (true) {
            final s = _find(buf, 0xFF, 0xD8, 0);
            if (s < 0) {
              if (buf.length > 4) buf.removeRange(0, buf.length - 1);
              break;
            }
            final e = _find(buf, 0xFF, 0xD9, s + 2);
            if (e < 0) {
              if (s > 0) buf.removeRange(0, s);
              break;
            }
            robotFrame.value = Uint8List.fromList(buf.sublist(s, e + 2));
            buf.removeRange(0, e + 2);
            if (!streamLive || streamError) {
              streamLive = true;
              streamError = false;
              notifyListeners();
            }
          }
          if (buf.length > 2 * 1024 * 1024) buf.clear();
        }
      } catch (_) {
        // unreachable / refused / bad status -> reported as "No camera detected!"
      } finally {
        client.close();
      }
      if (gen != _streamGen) break;
      streamLive = false;
      robotFrame.value = null;
      if (isConnected && !isMock) streamError = true;
      notifyListeners();
      await Future.delayed(const Duration(seconds: 2));
    }
  }

  // ------------------------------------------------------------ view toggles
  void toggleMirror() {
    mirrorCamera = !mirrorCamera;
    notifyListeners();
  }

  void toggleLandmarks() {
    showLandmarks = !showLandmarks;
    notifyListeners();
  }

  void togglePipSwap() {
    isPipSwapped = !isPipSwapped;
    notifyListeners();
  }

  void toggleSplitView() {
    isHorizontalSplit = !isHorizontalSplit;
    notifyListeners();
  }

  void setPracticeMode(bool value) {
    isPracticeMode = value;
    if (value) _burstStop(); // make sure the robot is stopped before practising
    notifyListeners();
  }

  void resetSensitivityDefaults() {
    gestureSensitivity = 0.70;
    maxMotorSpeed = 0.55;
    steeringTrim = 0.0;
    minPwmDeadband = 0.30;
    notifyListeners();
  }

  void updateSensitivity({
    double? gesture,
    double? speed,
    double? trim,
    double? deadband,
  }) {
    if (gesture != null) gestureSensitivity = gesture;
    if (speed != null) maxMotorSpeed = speed;
    if (trim != null) steeringTrim = trim;
    if (deadband != null) minPwmDeadband = deadband;
    notifyListeners();
  }

  @override
  void dispose() {
    _txTimer?.cancel();
    _beatTimer?.cancel();
    _watchTimer?.cancel();
    _stopStream();
    _socket?.close();
    _ctl.close();
    _disposeController(camController);
    super.dispose();
  }
}

// ============================================================================
// APP ROOT
// ============================================================================

class SteerApp extends StatefulWidget {
  final AppState appState;
  const SteerApp({super.key, required this.appState});

  @override
  State<SteerApp> createState() => _SteerAppState();
}

class _SteerAppState extends State<SteerApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    widget.appState.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Never leave the robot driving when the app is backgrounded.
    if (state == AppLifecycleState.resumed) {
      widget.appState.onAppResumed();
    } else {
      widget.appState.onAppPaused();
      if (state == AppLifecycleState.paused) {
        widget.appState.onAppBackgrounded();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'S.T.E.E.R.',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: SteerColors.bg,
        colorScheme: ColorScheme.fromSeed(seedColor: SteerColors.teal),
        useMaterial3: true,
      ),
      home: OnboardingScreen(appState: widget.appState),
    );
  }
}

// ============================================================================
// SHARED WIDGETS
// ============================================================================

class SteerBrandHeader extends StatelessWidget {
  final AppState appState;
  const SteerBrandHeader({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final on = appState.isConnected;
        return Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: SteerColors.teal,
                borderRadius: BorderRadius.circular(11),
              ),
              alignment: Alignment.center,
              child: const Text(
                'S',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w800,
                  fontSize: 18,
                ),
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              'S.T.E.E.R.',
              style: TextStyle(
                color: SteerColors.dark,
                fontWeight: FontWeight.w800,
                fontSize: 18,
                letterSpacing: 0.5,
              ),
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: on ? SteerColors.tealLight : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border:
                    Border.all(color: on ? SteerColors.teal : Colors.black12),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: on ? SteerColors.teal : Colors.grey,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    on
                        ? (appState.isMock ? 'Demo Robot Online' : 'ESP32 Online')
                        : 'Demo / Offline',
                    style: const TextStyle(
                      color: SteerColors.dark,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}

class SteerPageTopBar extends StatelessWidget {
  final String title;
  final Widget? trailing;

  const SteerPageTopBar({super.key, required this.title, this.trailing});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        InkWell(
          onTap: () => Navigator.of(context).pop(),
          borderRadius: BorderRadius.circular(14),
          child: Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
            ),
            alignment: Alignment.center,
            child: const Icon(
              Icons.chevron_left_rounded,
              color: SteerColors.dark,
              size: 26,
            ),
          ),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Text(
            title,
            style: const TextStyle(
              color: SteerColors.dark,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        if (trailing != null) trailing!,
      ],
    );
  }
}

IconData gestureIconFor(String code) {
  switch (code) {
    case '1':
      return Icons.touch_app_rounded;
    case '2':
      return Icons.pan_tool_alt_rounded;
    case '3':
      return Icons.back_hand_outlined;
    case '4':
      return Icons.front_hand_outlined;
    case 'Palm':
      return Icons.pan_tool_rounded;
    default:
      return Icons.warning_amber_rounded;
  }
}

/// Live phone camera (hand camera). Shows a spinner / permission prompt until
/// the hardware is ready and paints the 21-point hand skeleton on top:
///  - real plugin landmarks if AppState.handLandmarks is set,
///  - otherwise a skeleton anchored on the hand found by the built-in tracker,
///  - otherwise (only if [simulateIfIdle]) a gesture-driven demo skeleton.
class HandCameraView extends StatelessWidget {
  final AppState appState;
  final bool showOverlay;
  final bool simulateIfIdle;
  const HandCameraView({
    super.key,
    required this.appState,
    this.showOverlay = true,
    this.simulateIfIdle = false,
  });

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final c = appState.camController;
        // The plugin already mirrors the front camera preview, so we only
        // flip when the user's mirror setting differs from that default.
        final flip = appState.mirrorCamera != appState.isFrontCamera;
        final children = <Widget>[];

        if (c != null && c.value.isInitialized) {
          final portrait =
              MediaQuery.of(context).orientation == Orientation.portrait;
          final ar = c.value.aspectRatio;
          final w = portrait ? 100.0 : 100.0 * ar;
          final h = portrait ? 100.0 * ar : 100.0;
          Widget p = FittedBox(
            fit: BoxFit.cover,
            clipBehavior: Clip.hardEdge,
            child: SizedBox(width: w, height: h, child: CameraPreview(c)),
          );
          if (flip) {
            p = Transform(
              alignment: Alignment.center,
              transform: Matrix4.diagonal3Values(-1, 1, 1),
              child: p,
            );
          }
          children.add(Positioned.fill(child: ClipRect(child: p)));
        } else {
          children.add(Positioned.fill(
            child: CameraStatusPanel(
              status: appState.cameraStatus,
              message: appState.cameraError,
              onRetry: appState.retryCamera,
            ),
          ));
        }

        if (showOverlay && appState.showLandmarks) {
          final lm = appState.handLandmarks;
          final obs = appState.handObs;
          if (lm != null) {
            children.add(Positioned.fill(
              child: CustomPaint(painter: LandmarkOverlayPainter(lm, flip)),
            ));
          } else if (obs != null) {
            children.add(Positioned.fill(
              child: CustomPaint(
                painter: HandOverlayPainter(
                  template: buildHandTemplate(obs.gesture),
                  center: obs.center,
                  radiusNormH: obs.radiusNormH,
                  aspect: obs.aspect,
                  mirror: appState.mirrorCamera,
                ),
              ),
            ));
          } else if (simulateIfIdle &&
              !appState.trackingActive &&
              appState.activeGesture != 'LOST') {
            children.add(Positioned.fill(
              child: CustomPaint(
                painter: HandOverlayPainter(
                  template: buildHandTemplate(appState.activeGesture),
                  mirror: appState.mirrorCamera,
                ),
              ),
            ));
          }
        }
        return Stack(fit: StackFit.expand, children: children);
      },
    );
  }
}

/// Spinner while the camera starts, or a permission / error prompt with Retry.
class CameraStatusPanel extends StatelessWidget {
  final CameraStatus status;
  final String? message;
  final VoidCallback onRetry;
  const CameraStatusPanel({
    super.key,
    required this.status,
    required this.message,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final loading =
        status == CameraStatus.idle || status == CameraStatus.initializing;
    final denied = status == CameraStatus.denied;

    final title = loading
        ? 'Starting camera...'
        : denied
            ? 'Camera permission needed'
            : 'Camera unavailable';
    final body = message ??
        (denied
            ? 'Allow camera access for S.T.E.E.R. when prompted, or enable it in your phone/browser settings, then tap Retry.'
            : 'Showing simulated view.');

    return LayoutBuilder(
      builder: (context, c) {
        final compact = c.maxHeight < 220 || c.maxWidth < 180;

        final content = Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (loading)
              SizedBox(
                width: compact ? 20 : 28,
                height: compact ? 20 : 28,
                child: const CircularProgressIndicator(
                  strokeWidth: 2.5,
                  color: SteerColors.teal,
                ),
              )
            else
              Icon(
                denied
                    ? Icons.no_photography_outlined
                    : Icons.videocam_off_outlined,
                color: SteerColors.warning,
                size: compact ? 22 : 30,
              ),
            SizedBox(height: compact ? 6 : 10),
            Text(
              compact && !loading ? 'Tap to retry' : title,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: compact ? 11 : 15,
                fontWeight: FontWeight.w700,
              ),
            ),
            if (!compact && !loading) ...[
              const SizedBox(height: 6),
              Text(
                body,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: SteerColors.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                ),
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text('Retry'),
              ),
            ],
          ],
        );

        if (compact) {
          return InkWell(
            onTap: loading ? null : onRetry,
            child: Center(child: content),
          );
        }
        return Align(
          alignment: const Alignment(0, -0.7),
          child: Container(
            constraints: const BoxConstraints(maxWidth: 290),
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xE60B1E26),
              borderRadius: BorderRadius.circular(18),
            ),
            child: content,
          ),
        );
      },
    );
  }
}

/// Robot ESP32-CAM feed (MJPEG). Only shows video while a real ESP32 robot is
/// connected; otherwise (offline / demo) it shows a yellow warning.
class RobotFeedView extends StatelessWidget {
  final AppState appState;
  const RobotFeedView({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final robotOnline = appState.isConnected && !appState.isMock;
        return LayoutBuilder(
          builder: (context, c) {
            final compact = c.maxWidth < 180 || c.maxHeight < 180;
            if (!robotOnline) return _FeedMessage(compact: compact);
            return ValueListenableBuilder<Uint8List?>(
              valueListenable: appState.robotFrame,
              builder: (context, frame, _) {
                if (frame != null) {
                  return Image.memory(
                    frame,
                    fit: BoxFit.cover,
                    gaplessPlayback: true,
                    width: double.infinity,
                    height: double.infinity,
                    errorBuilder: (_, _, _) => _FeedMessage(compact: compact),
                  );
                }
                if (appState.streamError || appState.camStreamUrl.isEmpty) {
                  return _FeedMessage(compact: compact);
                }
                return _FeedMessage(compact: compact, connecting: true);
              },
            );
          },
        );
      },
    );
  }
}

class _FeedMessage extends StatelessWidget {
  final bool compact;
  final bool connecting;
  const _FeedMessage({required this.compact, this.connecting = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF16323E), Color(0xFF0B1E26)],
        ),
      ),
      alignment: Alignment.center,
      padding: const EdgeInsets.all(8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (connecting)
            SizedBox(
              width: compact ? 18 : 26,
              height: compact ? 18 : 26,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                color: SteerColors.teal,
              ),
            )
          else
            Icon(
              Icons.videocam_off_outlined,
              color: SteerColors.warningYellow,
              size: compact ? 22 : 32,
            ),
          SizedBox(height: compact ? 6 : 10),
          Text(
            connecting ? 'Connecting to ESP32-CAM...' : 'No camera detected!',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: connecting ? Colors.white70 : SteerColors.warningYellow,
              fontSize: compact ? 11 : 16,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 1: ONBOARDING
// ============================================================================

class OnboardingScreen extends StatelessWidget {
  final AppState appState;
  const OnboardingScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    final badges = ['1', '2', '3', '4', 'Palm'];

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SteerBrandHeader(appState: appState),
              const Spacer(flex: 2),
              RichText(
                text: const TextSpan(
                  style: TextStyle(
                    fontSize: 46,
                    fontWeight: FontWeight.w800,
                    color: SteerColors.dark,
                    height: 1.1,
                  ),
                  children: [
                    TextSpan(text: 'Steer with\nyour '),
                    TextSpan(
                      text: 'hand.',
                      style: TextStyle(color: SteerColors.teal),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Drive your robot with finger gestures. No joystick, no buttons.',
                style: TextStyle(
                  color: SteerColors.subtitle,
                  fontSize: 17,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 28),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: badges.map((b) {
                    return Container(
                      width: 56,
                      height: 62,
                      decoration: BoxDecoration(
                        color: SteerColors.tealLight,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(gestureIconFor(b),
                              size: 18, color: SteerColors.teal),
                          const SizedBox(height: 2),
                          Text(
                            b,
                            style: const TextStyle(
                              color: SteerColors.dark,
                              fontWeight: FontWeight.w800,
                              fontSize: 15,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
              const Spacer(flex: 3),
              SizedBox(
                width: double.infinity,
                height: 58,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: SteerColors.teal,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => HomeHubScreen(appState: appState),
                      ),
                    );
                  },
                  child: const Text(
                    'Get started',
                    style: TextStyle(
                      color: SteerColors.dark,
                      fontWeight: FontWeight.w800,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 2: HOME HUB
// ============================================================================

class HomeHubScreen extends StatelessWidget {
  final AppState appState;
  const HomeHubScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    void go(Widget w) =>
        Navigator.push(context, MaterialPageRoute(builder: (_) => w));

    final menuItems = [
      ('1', 'Drive', Icons.gamepad_outlined, () {
        appState.setPracticeMode(false);
        go(DriveScreen(appState: appState));
      }),
      ('2', 'Connect robot', Icons.wifi_tethering_rounded,
          () => go(ConnectRobotScreen(appState: appState))),
      ('3', 'Sensitivity', Icons.tune_rounded,
          () => go(SensitivityScreen(appState: appState))),
      ('4', 'FPV camera', Icons.videocam_outlined,
          () => go(FpvCameraScreen(appState: appState))),
      ('5', 'Gestures', Icons.front_hand_outlined,
          () => go(GesturesScreen(appState: appState))),
      ('6', 'About', Icons.info_outline_rounded,
          () => go(AboutScreen(appState: appState))),
    ];

    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SteerBrandHeader(appState: appState),
                  const SizedBox(height: 22),
                  const Text(
                    'What now?',
                    style: TextStyle(
                      color: SteerColors.dark,
                      fontSize: 32,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Expanded(
                    child: GridView.builder(
                      itemCount: menuItems.length,
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        mainAxisSpacing: 14,
                        crossAxisSpacing: 14,
                        childAspectRatio: 1.05,
                      ),
                      itemBuilder: (context, index) {
                        final item = menuItems[index];
                        return InkWell(
                          onTap: item.$4,
                          borderRadius: BorderRadius.circular(24),
                          child: Container(
                            padding: const EdgeInsets.all(18),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(24),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Container(
                                      width: 44,
                                      height: 44,
                                      decoration: BoxDecoration(
                                        color: SteerColors.tealLight,
                                        borderRadius: BorderRadius.circular(14),
                                      ),
                                      alignment: Alignment.center,
                                      child: Text(
                                        item.$1,
                                        style: const TextStyle(
                                          color: SteerColors.teal,
                                          fontWeight: FontWeight.w800,
                                          fontSize: 17,
                                        ),
                                      ),
                                    ),
                                    Icon(item.$3,
                                        color: SteerColors.subtitle, size: 20),
                                  ],
                                ),
                                Text(
                                  item.$2,
                                  style: const TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 18,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  if (appState.isMoving)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: SizedBox(
                        width: double.infinity,
                        height: 52,
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: SteerColors.danger,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(18),
                            ),
                          ),
                          onPressed: appState.emergencyStop,
                          icon: const Icon(Icons.stop_circle_outlined),
                          label: Text(
                            'EMERGENCY STOP (${appState.commandName} active)',
                            style: const TextStyle(fontWeight: FontWeight.w800),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 3: CONNECT ROBOT
// ============================================================================

class ConnectRobotScreen extends StatefulWidget {
  final AppState appState;
  const ConnectRobotScreen({super.key, required this.appState});

  @override
  State<ConnectRobotScreen> createState() => _ConnectRobotScreenState();
}

class _ConnectRobotScreenState extends State<ConnectRobotScreen> {
  late TextEditingController _ipController;
  late TextEditingController _portController;
  late TextEditingController _camUrlController;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: widget.appState.robotIp);
    _portController = TextEditingController(text: widget.appState.udpPort);
    _camUrlController =
        TextEditingController(text: widget.appState.camStreamUrl);
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    _camUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.appState;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final okColor = s.isConnected ? SteerColors.teal : const Color(0xFF8B1D1D);
        final statusText = s.isConnected
            ? (s.isMock
                ? 'Demo link active (simulated robot)'
                : 'Connected to ${s.robotIp}:${s.udpPort}  -  ${s.latencyMs} ms')
            : (s.statusMessage ?? 'Robot not found. Check address');

        return Scaffold(
          body: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SteerPageTopBar(
                    title: 'Connect robot',
                    trailing: TextButton.icon(
                      onPressed: s.isScanning
                          ? null
                          : () async {
                              await s.scanNetwork();
                              _ipController.text = s.robotIp;
                              _camUrlController.text = s.camStreamUrl;
                            },
                      icon: s.isScanning
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.radar_rounded, size: 18),
                      label: Text(s.isScanning ? 'Scanning...' : 'Auto-Scan'),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Robot address on the same network as this phone. Leave empty for camera-only demo.',
                          style: TextStyle(
                            color: SteerColors.subtitle,
                            fontSize: 15,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              flex: 3,
                              child: TextField(
                                controller: _ipController,
                                keyboardType: TextInputType.number,
                                style: const TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w600,
                                  color: SteerColors.dark,
                                ),
                                decoration: _fieldDecoration(
                                  'ESP32 IP Address',
                                  '192.168.4.1',
                                  highlight: true,
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              flex: 1,
                              child: TextField(
                                controller: _portController,
                                keyboardType: TextInputType.number,
                                style: const TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w600,
                                  color: SteerColors.dark,
                                ),
                                decoration: _fieldDecoration('HTTP Port', '80'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          height: 54,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.teal,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: s.isTesting
                                ? null
                                : () => s.testConnection(
                                      _ipController.text,
                                      _portController.text,
                                      _camUrlController.text,
                                    ),
                            child: s.isTesting
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2, color: Colors.white),
                                  )
                                : const Text(
                                    'Save and test',
                                    style: TextStyle(
                                      color: SteerColors.dark,
                                      fontWeight: FontWeight.w800,
                                      fontSize: 17,
                                    ),
                                  ),
                          ),
                        ),
                        const SizedBox(height: 14),
                        if (s.hasTestedConnection)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 10),
                            decoration: BoxDecoration(
                              color: s.isConnected
                                  ? SteerColors.tealLight
                                  : SteerColors.dangerLight,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  s.isConnected
                                      ? Icons.check_circle_rounded
                                      : Icons.error_outline_rounded,
                                  size: 18,
                                  color: okColor,
                                ),
                                const SizedBox(width: 8),
                                Flexible(
                                  child: Text(
                                    statusText,
                                    style: TextStyle(
                                      color: s.isConnected
                                          ? SteerColors.dark
                                          : okColor,
                                      fontWeight: FontWeight.w700,
                                      fontSize: 14,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        const SizedBox(height: 8),
                        SwitchListTile(
                          contentPadding: EdgeInsets.zero,
                          activeThumbColor: SteerColors.teal,
                          title: const Text('Demo mode (simulate robot)',
                              style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  color: SteerColors.dark)),
                          subtitle: Text(kIsWeb
                              ? 'Always on in the browser (cannot reach the robot)'
                              : 'No commands are sent over Wi-Fi'),
                          value: s.isMock,
                          onChanged: kIsWeb ? null : s.setMockMode,
                        ),
                        SwitchListTile(
                          contentPadding: EdgeInsets.zero,
                          activeThumbColor: SteerColors.teal,
                          title: const Text('Send even without status reply',
                              style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  color: SteerColors.dark)),
                          subtitle: const Text(
                              'Send commands even if /status gets no reply'),
                          value: s.oneWayMode,
                          onChanged: s.isMock ? null : s.setOneWayMode,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Optional: add the ESP32-CAM address to turn on live FPV video.',
                          style: TextStyle(
                            color: SteerColors.subtitle,
                            fontSize: 15,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 14),
                        TextField(
                          controller: _camUrlController,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                            color: SteerColors.dark,
                          ),
                          decoration: InputDecoration(
                            labelText: 'ESP32-CAM MJPEG Stream URL',
                            hintText: 'http://192.168.4.2:81/stream',
                            prefixIcon: const Icon(Icons.videocam_outlined),
                            contentPadding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 14),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  InputDecoration _fieldDecoration(String label, String hint,
      {bool highlight = false}) {
    return InputDecoration(
      labelText: label,
      hintText: hint,
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: BorderSide(
          color: highlight ? SteerColors.teal : Colors.black26,
          width: highlight ? 1.5 : 1,
        ),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: SteerColors.teal, width: 2),
      ),
    );
  }
}

// ============================================================================
// SCREEN 4: DRIVE
// ============================================================================

class DriveScreen extends StatefulWidget {
  final AppState appState;
  const DriveScreen({super.key, required this.appState});

  @override
  State<DriveScreen> createState() => _DriveScreenState();
}

class _DriveScreenState extends State<DriveScreen> {
  AppState get appState => widget.appState;

  @override
  void initState() {
    super.initState();
    appState.acquireCamera();
    appState.acquireTracking(); // live hand detection (Drive AND Practice)
  }

  @override
  void dispose() {
    appState.sendStopBurst(); // never leave the robot moving
    appState.releaseTracking();
    appState.releaseCamera();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const controls = [
      ('1', 'FWD'),
      ('2', 'BACK'),
      ('3', 'LEFT'),
      ('4', 'RIGHT'),
      ('Palm', 'STOP'),
    ];

    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final isHandLost = appState.activeGesture == 'LOST';
        final hasRealLandmarks =
            appState.handLandmarks != null || appState.handObs != null;

        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(
                    title: appState.isPracticeMode ? 'Practice Drive' : 'Drive',
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Toggle Hand Skeleton Overlay',
                          onPressed: appState.toggleLandmarks,
                          icon: Icon(
                            Icons.hub_outlined,
                            color: appState.showLandmarks
                                ? SteerColors.teal
                                : SteerColors.subtitle,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Mirror Camera Feed',
                          onPressed: appState.toggleMirror,
                          icon: Icon(
                            Icons.flip_rounded,
                            color: appState.mirrorCamera
                                ? SteerColors.teal
                                : SteerColors.subtitle,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Switch Front/Rear Camera',
                          onPressed: appState.toggleFrontCamera,
                          icon: const Icon(Icons.cameraswitch_outlined,
                              color: SteerColors.dark),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(28),
                      child: Container(
                        width: double.infinity,
                        color: SteerColors.cameraBg,
                        child: Stack(
                          children: [
                            Positioned.fill(
                              child: HandCameraView(
                                appState: appState,
                                simulateIfIdle: true,
                              ),
                            ),
                            Positioned(
                              top: 16,
                              left: 16,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 14, vertical: 7),
                                decoration: BoxDecoration(
                                  color: isHandLost
                                      ? SteerColors.warning
                                      : SteerColors.teal,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  isHandLost
                                      ? 'Hand Lost - Auto-Brake'
                                      : appState.isPracticeMode
                                          ? 'Practice mode (Motors Off)'
                                          : 'Gesture mode',
                                  style: const TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 13,
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              top: 18,
                              right: 16,
                              child: Text(
                                '${appState.isFrontCamera ? "Front Cam" : "Rear Cam"}${appState.mirrorCamera ? " (Mirrored)" : ""}',
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                            Center(
                              child: CustomPaint(
                                painter: DashedFramePainter(),
                                child: SizedBox(
                                  width: 210,
                                  height: 250,
                                  child: Center(
                                    child: Text(
                                      isHandLost
                                          ? 'No hand in frame\nMotors Locked'
                                          : (hasRealLandmarks ||
                                                  appState.cameraStatus !=
                                                      CameraStatus.ready
                                              ? ''
                                              : 'Show your hand'),
                                      textAlign: TextAlign.center,
                                      style: TextStyle(
                                        color: isHandLost
                                            ? SteerColors.warning
                                            : Colors.white70,
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              bottom: 12,
                              right: 14,
                              child: TextButton.icon(
                                style: TextButton.styleFrom(
                                  backgroundColor: Colors.black38,
                                  foregroundColor: Colors.white70,
                                ),
                                onPressed: () => appState
                                    .setGesture(isHandLost ? 'Palm' : 'LOST'),
                                icon: Icon(
                                  isHandLost
                                      ? Icons.visibility_rounded
                                      : Icons.visibility_off_outlined,
                                  size: 16,
                                ),
                                label: Text(
                                  isHandLost
                                      ? 'Restore Hand'
                                      : 'Simulate Hand Lost',
                                  style: const TextStyle(fontSize: 12),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Flexible(
                              child: Text(
                                appState.commandName,
                                overflow: TextOverflow.ellipsis,
                                style: TextStyle(
                                  color: isHandLost
                                      ? SteerColors.danger
                                      : SteerColors.dark,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                            Text(
                              'Link ${appState.linkQuality}%  Battery ${appState.batteryLevel}%',
                              style: const TextStyle(
                                color: SteerColors.subtitle,
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: ValueListenableBuilder<String>(
                            valueListenable: appState.packetNotifier,
                            builder: (_, p, _) => Text(
                              appState.isPracticeMode
                                  ? 'Practice: motors off, nothing is sent'
                                  : (appState.isConnected
                                      ? 'TX ${appState.isMock ? "(demo) " : ""}$p'
                                      : 'Not connected - commands are not sent'),
                              style: const TextStyle(
                                color: SteerColors.subtitle,
                                fontSize: 12,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Text(
                            appState.trackingLabel,
                            style: TextStyle(
                              color: appState.handObs != null
                                  ? SteerColors.teal
                                  : SteerColors.subtitle,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: controls.map((item) {
                            final isSelected =
                                appState.activeGesture == item.$1 &&
                                    !appState.estop ||
                                (appState.estop && item.$1 == 'Palm');
                            return GestureDetector(
                              onTap: () => appState.setGesture(item.$1),
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 160),
                                width: 58,
                                height: 68,
                                decoration: BoxDecoration(
                                  color: isSelected
                                      ? SteerColors.teal
                                      : const Color(0xFFF1F5F5),
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Text(
                                      item.$1,
                                      style: const TextStyle(
                                        color: SteerColors.dark,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 17,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      item.$2,
                                      style: const TextStyle(
                                        color: SteerColors.dark,
                                        fontWeight: FontWeight.w700,
                                        fontSize: 11,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          }).toList(),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) =>
                                      FpvCameraScreen(appState: appState),
                                ),
                              );
                            },
                            child: const Text(
                              'FPV',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.danger,
                              foregroundColor: Colors.white,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: appState.emergencyStop,
                            child: const Text(
                              'STOP',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 5: FPV CAMERA
// ============================================================================

class FpvCameraScreen extends StatefulWidget {
  final AppState appState;
  const FpvCameraScreen({super.key, required this.appState});

  @override
  State<FpvCameraScreen> createState() => _FpvCameraScreenState();
}

class _FpvCameraScreenState extends State<FpvCameraScreen> {
  AppState get appState => widget.appState;

  @override
  void initState() {
    super.initState();
    appState.acquireCamera();
    appState.acquireStream();
  }

  @override
  void dispose() {
    appState.releaseCamera();
    appState.releaseStream();
    super.dispose();
  }

  Widget _box(Widget child, String badge) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Container(
        width: double.infinity,
        color: SteerColors.cameraBg,
        child: Stack(
          fit: StackFit.expand,
          children: [
            child,
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  badge,
                  style: const TextStyle(
                    color: SteerColors.dark,
                    fontWeight: FontWeight.w800,
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final robot = RobotFeedView(appState: appState);
        final hand = HandCameraView(
            appState: appState, showOverlay: appState.showLandmarks);
        final main = appState.isPipSwapped ? hand : robot;
        final pip = appState.isPipSwapped ? robot : hand;
        final robotBadge = appState.streamLive ? 'Live' : 'Robot (offline)';

        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(
                    title: 'FPV camera',
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Toggle Horizontal Split / PiP View',
                          onPressed: appState.toggleSplitView,
                          icon: Icon(
                            appState.isHorizontalSplit
                                ? Icons.picture_in_picture_alt_rounded
                                : Icons.vertical_split_rounded,
                            color: SteerColors.teal,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Switch Front/Rear Hand Cam',
                          onPressed: appState.toggleFrontCamera,
                          icon: const Icon(Icons.cameraswitch_outlined,
                              color: SteerColors.dark),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: appState.isHorizontalSplit
                        ? Column(
                            children: [
                              Expanded(child: _box(robot, robotBadge)),
                              const SizedBox(height: 10),
                              Expanded(
                                child: _box(
                                  hand,
                                  appState.isFrontCamera ? 'Front' : 'Rear',
                                ),
                              ),
                            ],
                          )
                        : Stack(
                            children: [
                              Positioned.fill(
                                child: _box(
                                  main,
                                  appState.isPipSwapped ? 'Hand' : robotBadge,
                                ),
                              ),
                              Positioned(
                                bottom: 18,
                                right: 18,
                                child: GestureDetector(
                                  onTap: appState.togglePipSwap,
                                  child: Container(
                                    width: 115,
                                    height: 145,
                                    decoration: BoxDecoration(
                                      color: const Color(0xFF0B1E26),
                                      borderRadius: BorderRadius.circular(20),
                                      border: Border.all(
                                          color: SteerColors.teal, width: 2),
                                    ),
                                    child: ClipRRect(
                                      borderRadius: BorderRadius.circular(18),
                                      child: Stack(
                                        fit: StackFit.expand,
                                        children: [
                                          pip,
                                          const Positioned(
                                            top: 6,
                                            right: 6,
                                            child: Icon(
                                              Icons.swap_horiz_rounded,
                                              color: SteerColors.teal,
                                              size: 22,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                  ),
                  const SizedBox(height: 14),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(22),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('Command',
                                style: TextStyle(
                                    color: SteerColors.subtitle, fontSize: 13)),
                            const SizedBox(height: 2),
                            Text(
                              appState.commandName,
                              style: const TextStyle(
                                color: SteerColors.teal,
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            const Text('Latency',
                                style: TextStyle(
                                    color: SteerColors.subtitle, fontSize: 13)),
                            const SizedBox(height: 2),
                            Text(
                              appState.isConnected
                                  ? '${appState.latencyMs} ms'
                                  : '-- ms',
                              style: const TextStyle(
                                color: SteerColors.dark,
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: SteerColors.danger,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      onPressed: appState.emergencyStop,
                      child: const Text(
                        'STOP',
                        style:
                            TextStyle(fontWeight: FontWeight.w800, fontSize: 17),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 6: GESTURES GUIDE
// ============================================================================

class GesturesScreen extends StatelessWidget {
  final AppState appState;
  const GesturesScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    const gestures = [
      ('1', 'Forward', 'One finger up'),
      ('2', 'Backward', 'Two fingers up'),
      ('3', 'Turn left', 'Three fingers up'),
      ('4', 'Turn right', 'Four fingers up'),
      ('Palm', 'Stop', 'Open palm or fist, stops at once'),
    ];

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
          child: Column(
            children: [
              const SteerPageTopBar(title: 'Gestures'),
              const SizedBox(height: 18),
              Expanded(
                child: ListView.separated(
                  itemCount: gestures.length,
                  separatorBuilder: (_, _) => const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final g = gestures[index];
                    return Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(22),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 56,
                            height: 56,
                            decoration: BoxDecoration(
                              color: SteerColors.tealLight,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(gestureIconFor(g.$1),
                                    size: 18, color: SteerColors.teal),
                                Text(
                                  g.$1,
                                  style: const TextStyle(
                                    color: SteerColors.teal,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 14,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  g.$2,
                                  style: const TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 18,
                                  ),
                                ),
                                const SizedBox(height: 3),
                                Text(
                                  g.$3,
                                  style: const TextStyle(
                                    color: SteerColors.subtitle,
                                    fontSize: 14,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: SteerColors.teal,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(18),
                    ),
                  ),
                  onPressed: () {
                    appState.setPracticeMode(true);
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => DriveScreen(appState: appState),
                      ),
                    );
                  },
                  child: const Text(
                    'Practice mode',
                    style: TextStyle(
                      color: SteerColors.dark,
                      fontWeight: FontWeight.w800,
                      fontSize: 17,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 7: SENSITIVITY & CALIBRATION
// ============================================================================

class SensitivityScreen extends StatelessWidget {
  final AppState appState;
  const SensitivityScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  const SteerPageTopBar(title: 'Sensitivity'),
                  const SizedBox(height: 18),
                  Expanded(
                    child: ListView(
                      children: [
                        _buildSliderCard(
                          title: 'Gesture detection',
                          valueLabel:
                              '${(appState.gestureSensitivity * 100).round()}%',
                          subtitle: 'Higher reacts faster but may misread.',
                          value: appState.gestureSensitivity,
                          min: 0.1,
                          max: 1.0,
                          onChanged: (v) =>
                              appState.updateSensitivity(gesture: v),
                        ),
                        const SizedBox(height: 14),
                        _buildSliderCard(
                          title: 'Motor speed',
                          valueLabel:
                              '${(appState.maxMotorSpeed * 100).round()}%',
                          subtitle: 'Top speed the robot will drive at.',
                          value: appState.maxMotorSpeed,
                          min: 0.2,
                          max: 1.0,
                          onChanged: (v) =>
                              appState.updateSensitivity(speed: v),
                        ),
                        const SizedBox(height: 14),
                        _buildSliderCard(
                          title: 'Steering trim (Left / Right balance)',
                          valueLabel: appState.steeringTrim == 0
                              ? '0 (Center)'
                              : (appState.steeringTrim > 0
                                  ? '+${appState.steeringTrim.round()} R'
                                  : '${appState.steeringTrim.round()} L'),
                          subtitle:
                              'Offsets left/right motor power so the 2WD chassis drives straight.',
                          value: appState.steeringTrim,
                          min: -10.0,
                          max: 10.0,
                          onChanged: (v) => appState.updateSensitivity(
                            trim: v.roundToDouble(),
                          ),
                        ),
                        const SizedBox(height: 14),
                        _buildSliderCard(
                          title: 'Minimum PWM deadband',
                          valueLabel:
                              '${(appState.minPwmDeadband * 100).round()}%',
                          subtitle:
                              'Minimum starting power required to overcome DC gearbox friction.',
                          value: appState.minPwmDeadband,
                          min: 0.15,
                          max: 0.50,
                          onChanged: (v) =>
                              appState.updateSensitivity(deadband: v),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: appState.resetSensitivityDefaults,
                            child: const Text(
                              'Default',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.teal,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: () {
                              appState.saveSettings();
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text('Calibration settings saved.'),
                                  duration: Duration(seconds: 1),
                                ),
                              );
                              Navigator.pop(context);
                            },
                            child: const Text(
                              'Save',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildSliderCard({
    required String title,
    required String valueLabel,
    required String subtitle,
    required double value,
    required double min,
    required double max,
    required ValueChanged<double> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    color: SteerColors.dark,
                    fontWeight: FontWeight.w800,
                    fontSize: 17,
                  ),
                ),
              ),
              Text(
                valueLabel,
                style: const TextStyle(
                  color: SteerColors.teal,
                  fontWeight: FontWeight.w800,
                  fontSize: 17,
                ),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: SteerColors.teal,
              inactiveTrackColor: const Color(0xFFE4ECEB),
              thumbColor: SteerColors.teal,
              trackHeight: 8,
              overlayColor: SteerColors.teal.withValues(alpha: 0.15),
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              onChanged: onChanged,
            ),
          ),
          Text(
            subtitle,
            style: const TextStyle(color: SteerColors.subtitle, fontSize: 14),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 8: ABOUT
// ============================================================================

class AboutScreen extends StatelessWidget {
  final AppState appState;
  const AboutScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SteerPageTopBar(title: 'About S.T.E.E.R.'),
              const SizedBox(height: 18),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Smart Teleoperated ESP32 Educational Robot',
                      style: TextStyle(
                        color: SteerColors.dark,
                        fontWeight: FontWeight.w800,
                        fontSize: 18,
                      ),
                    ),
                    SizedBox(height: 10),
                    Text(
                      '- Vision Pipeline: 21-point MediaPipe Hand Landmarks\n'
                      '- Transport: HTTP GET /move over local Wi-Fi (on change + every 300 ms)\n'
                      '- Request: /move?left=1&right=-1&speed=55\n'
                      '- Heartbeat: app polls /status every second\n'
                      '- Failsafe: zero-PWM brake when hand leaves frame\n'
                      '- Hardware: ESP32 + L298N / TB6612FNG 2WD chassis + optional ESP32-CAM',
                      style: TextStyle(
                        color: SteerColors.subtitle,
                        fontSize: 15,
                        height: 1.6,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// PAINTERS
// ============================================================================

/// Dashed guide frame: keep your hand inside it.
class DashedFramePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final borderPaint = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final rrect = RRect.fromRectAndRadius(
      Offset.zero & size,
      const Radius.circular(28),
    );
    final path = Path()..addRRect(rrect);
    for (final metric in path.computeMetrics()) {
      double distance = 0;
      while (distance < metric.length) {
        final next = math.min(distance + 8, metric.length);
        canvas.drawPath(metric.extractPath(distance, next), borderPaint);
        distance += 14;
      }
    }
  }

  @override
  bool shouldRepaint(covariant DashedFramePainter old) => false;
}

/// Paints a 21-point hand skeleton. With [center] it is anchored on the
/// detected hand (mapped through the same "cover" crop as the camera preview);
/// without it, it is drawn at the middle of the viewport as a demo.
class HandOverlayPainter extends CustomPainter {
  final List<Offset> template; // palm-radius units
  final Offset? center; // normalized upright-image coordinates
  final double radiusNormH;
  final double aspect; // upright image width / height
  final bool mirror;

  HandOverlayPainter({
    required this.template,
    this.center,
    this.radiusNormH = 0.1,
    this.aspect = 0.75,
    required this.mirror,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (template.length != 21) return;
    final c = center;
    late Offset origin;
    late double unit;
    if (c == null) {
      origin = Offset(size.width * 0.5, size.height * 0.55);
      unit = math.min(size.width, size.height) * 0.16;
    } else {
      final scale = math.max(size.width / aspect, size.height);
      final drawnW = aspect * scale;
      final drawnH = scale;
      final ox = (size.width - drawnW) / 2;
      final oy = (size.height - drawnH) / 2;
      final nx = mirror ? 1 - c.dx : c.dx;
      origin = Offset(ox + nx * drawnW, oy + c.dy * drawnH);
      unit = radiusNormH * drawnH;
    }
    unit = math.max(8.0, math.min(140.0, unit));

    Offset p(Offset t) =>
        origin + Offset((mirror ? -t.dx : t.dx) * unit, t.dy * unit);

    final line = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.85)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    final dot = Paint()..color = Colors.white;
    for (final e in kHandConnections) {
      canvas.drawLine(p(template[e[0]]), p(template[e[1]]), line);
    }
    for (final t in template) {
      canvas.drawCircle(p(t), 3.4, dot);
    }
  }

  @override
  bool shouldRepaint(covariant HandOverlayPainter old) => true;
}

/// Draws real 21-point landmarks (normalized) over the whole camera viewport.
class LandmarkOverlayPainter extends CustomPainter {
  final List<Offset> landmarks;
  final bool flip;
  LandmarkOverlayPainter(this.landmarks, this.flip);

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.length != 21) return;
    Offset p(Offset n) =>
        Offset((flip ? 1 - n.dx : n.dx) * size.width, n.dy * size.height);
    final line = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.8)
      ..strokeWidth = 2.5;
    final dot = Paint()..color = Colors.white;
    for (final c in kHandConnections) {
      canvas.drawLine(p(landmarks[c[0]]), p(landmarks[c[1]]), line);
    }
    for (final l in landmarks) {
      canvas.drawCircle(p(l), 3.5, dot);
    }
  }

  @override
  bool shouldRepaint(covariant LandmarkOverlayPainter old) => true;
}
