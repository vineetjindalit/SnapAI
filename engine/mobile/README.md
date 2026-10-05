# Snappy Mobile App (React Native scaffold)

Bare-bones structure ready for sprint-3 buildout. Targets iOS first,
Android via React Native's shared codebase (~80% reuse).

## Status

This directory contains the **scaffold and core screens only**. A
production app is ~3 weeks of focused mobile work. What's here:

- Project structure + navigation skeleton
- Core API client (`src/api.ts`) wired against the existing server
- WebSocket client for live streaming
- Screen stubs: Auth, EventList, LiveCapture, Album
- TypeScript config + ESLint baseline

## Bootstrap

```bash
cd mobile
npx react-native@latest init SnappyApp --template react-native-template-typescript
# (or use Expo: npx create-expo-app SnappyApp --template tabs)
cp -r src/* SnappyApp/src/    # drop the scaffolded files in
cd SnappyApp
npm install zustand react-query @react-navigation/native @react-navigation/stack \
            react-native-vision-camera @react-native-async-storage/async-storage
```

Then:

```bash
# iOS
cd ios && pod install && cd ..
npx react-native run-ios

# Android
npx react-native run-android
```

## What still needs to be built (3 weeks)

| Week | Scope |
|---|---|
| 1 | Camera screen with VisionCamera + WebSocket streaming wired up |
| 2 | Album viewer + share sheet + push notifications (FCM/APNs) |
| 3 | Login flow + onboarding wizard + TestFlight build |

See `OWNERSHIP.md` sprint 3 for the detailed task list.

## Configuration

`SnappyApp/src/config.ts`:

```ts
export const SNAPPY_API_BASE = __DEV__
  ? "http://localhost:8765"
  : "https://api.snappy.example.com";
```
