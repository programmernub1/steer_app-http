#!/usr/bin/env python3
"""Prepare an Android Flutter project for S.T.E.E.R.

Run from the Flutter project root:   python3 tools/patch_android.py

 - adds INTERNET + CAMERA permissions (without CAMERA the phone camera never starts)
 - allows plain-http traffic (ESP32-CAM MJPEG stream)
 - raises minSdk to 24 (required by the camera plugin)
Safe to run more than once.
"""
import pathlib
import re
import sys

root = pathlib.Path('.')
manifest = root / 'android/app/src/main/AndroidManifest.xml'
if not manifest.exists():
    sys.exit('AndroidManifest.xml not found. Run this from the Flutter project root '
             '(create the android folder with: flutter create --platforms=android .)')

m = manifest.read_text()
changed = []

perms = [
    '<uses-permission android:name="android.permission.INTERNET"/>',
    '<uses-permission android:name="android.permission.CAMERA"/>',
    '<uses-feature android:name="android.hardware.camera" android:required="false"/>',
]
missing = [p for p in perms if p.split('"')[1] not in m]
if missing:
    m = re.sub(r'(<manifest[^>]*>)', lambda g: g.group(1) + '\n    ' + '\n    '.join(missing),
               m, count=1)
    changed.append('permissions: ' + ', '.join(p.split('"')[1].split('.')[-1] for p in missing))

if 'usesCleartextTraffic' not in m:
    m = m.replace('<application', '<application android:usesCleartextTraffic="true"', 1)
    changed.append('usesCleartextTraffic')

manifest.write_text(m)

for name in ('android/app/build.gradle.kts', 'android/app/build.gradle'):
    g = root / name
    if not g.exists():
        continue
    t = g.read_text()
    new = re.sub(r'minSdk(Version)?(\s*=\s*|\s+)flutter\.minSdkVersion',
                 lambda x: 'minSdk' + ('Version' if x.group(1) else '') + x.group(2) + '24', t)
    new = re.sub(r'(minSdk(?:Version)?(?:\s*=\s*|\s+))(\d+)',
                 lambda x: x.group(1) + ('24' if int(x.group(2)) < 24 else x.group(2)), new)
    if new != t:
        g.write_text(new)
        changed.append(f'minSdk -> 24 ({name})')

print('Patched: ' + '; '.join(changed) if changed else 'Nothing to change - already set up.')
