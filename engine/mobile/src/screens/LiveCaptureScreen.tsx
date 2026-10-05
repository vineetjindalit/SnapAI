// mobile/src/screens/LiveCaptureScreen.tsx
//
// Core screen: native camera frames → server WS → live capture HUD.
// This is the scaffold; sprint-3 wires it into the navigation stack
// and adds the polished UI elements (recent captures strip, ensemble
// signals panel, feedback buttons).

import React, { useEffect, useRef, useState } from "react";
import { Camera, useCameraDevice, useCameraPermission } from "react-native-vision-camera";
import { Sessions, frameStreamUrl } from "../api";
import { StyleSheet, Text, View, Button } from "react-native";

type Props = { sid: string };

export default function LiveCaptureScreen({ sid }: Props) {
  const device = useCameraDevice("back");
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [captures, setCaptures] = useState<number>(0);
  const [moment, setMoment]     = useState<string>("");
  const [score, setScore]       = useState<number>(0);
  const [emotion, setEmotion]   = useState<string>("neutral");

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    const ws = new WebSocket(frameStreamUrl(sid));
    ws.onopen   = () => console.log("ws open");
    ws.onerror  = (e) => console.error("ws error", e);
    ws.onmessage = (msg) => {
      try {
        const d = JSON.parse(msg.data);
        if (d.type !== "frame_result") return;
        setCaptures(d.total_captures || 0);
        setMoment(d.moment?.detected_moment || "");
        setScore(d.analysis?.total_score || 0);
        setEmotion(d.emotion?.dominant || "neutral");
      } catch {}
    };
    wsRef.current = ws;
    return () => ws.close();
  }, [sid]);

  // ── Frame send loop ─────────────────────────────────────────────────
  // Every 200ms, take a snapshot and send to the server. VisionCamera v3
  // exposes `takePhoto()`; we use that to keep the scaffold simple.
  useEffect(() => {
    if (!cameraRef.current) return;
    const t = setInterval(async () => {
      try {
        if (!cameraRef.current || wsRef.current?.readyState !== WebSocket.OPEN) return;
        const photo = await cameraRef.current.takePhoto({
          flash: "off", qualityPrioritization: "speed",
        });
        // photo.path is a local file URI. Read + base64 encode (lib needed: react-native-fs)
        // Sketch only — sprint 3 wires this up properly.
        // const b64 = await RNFS.readFile(photo.path, "base64");
        // wsRef.current.send(JSON.stringify({ frame: b64 }));
      } catch (e) {
        console.error("capture loop error", e);
      }
    }, 200);
    return () => clearInterval(t);
  }, [cameraRef.current]);

  if (!hasPermission) return <Text style={styles.center}>Camera permission required</Text>;
  if (!device) return <Text style={styles.center}>No camera device</Text>;

  return (
    <View style={styles.root}>
      <Camera style={StyleSheet.absoluteFill} device={device} isActive={true}
              ref={cameraRef} photo={true} />
      <View style={styles.hud}>
        <Text style={styles.hudTitle}>Snappy · live</Text>
        <Text style={styles.hudLine}>📸 {captures} captured</Text>
        <Text style={styles.hudLine}>🎯 {moment || "—"} · score {Math.round(score * 100)}</Text>
        <Text style={styles.hudLine}>😀 {emotion}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, textAlign: "center", textAlignVertical: "center", color: "#fff" },
  hud:    { position: "absolute", top: 40, left: 16, padding: 12,
            backgroundColor: "rgba(0,0,0,0.55)", borderRadius: 8 },
  hudTitle:{ color: "#fff", fontWeight: "700", fontSize: 12, marginBottom: 6 },
  hudLine: { color: "#fff", fontSize: 14, marginVertical: 2 },
});
