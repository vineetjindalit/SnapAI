// savePhotos.ts — get album photos OFF the server and ONTO the user's device.
//
// The ONLY way JS can put a file into a phone's camera roll is the native
// Share Sheet (navigator.share with files) — browsers deliberately block any
// silent write to the photo library. That call is gated on "user activation":
// it must fire close to the original tap, or mobile Safari silently refuses
// it. FETCHING PHOTOS ONE AT A TIME (await in a loop) was slow enough to burn
// through that window — share() then failed silently, and the old fallback
// (`<a download>` clicks) doesn't actually save anything on mobile Safari
// either, so nothing reached the phone; the photos just stayed where they've
// always lived, on the server.
//
// Fix: fetch every photo IN PARALLEL (fast, stays inside the activation
// window) and call share() immediately once they're ready. On desktop, or if
// share isn't available, fall back to a real `<a download>` (which DOES work
// there). On mobile without share support, don't pretend a download worked —
// tell the caller so it can point the user at the one thing that reliably
// works everywhere: long-press a photo → Save Image.

export const isMobile = () =>
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 0 && /Mac/i.test(navigator.userAgent)); // iPadOS reports as Mac

async function fetchFiles(urls: string[]): Promise<File[]> {
  const results = await Promise.all(urls.map(async (u) => {
    try {
      const res = await fetch(u);
      if (!res.ok) return null;
      const blob = await res.blob();
      const name = (u.split("/").pop() || "snapai.jpg").split("?")[0];
      // Force a real image MIME type — some browsers refuse to share() a
      // file whose type isn't recognized, and an octet-stream blob would
      // silently fail canShare().
      const type = blob.type && blob.type.startsWith("image/") ? blob.type : "image/jpeg";
      return new File([blob], name, { type });
    } catch { return null; }               // skip a photo that fails to load
  }));
  return results.filter((f): f is File => f !== null);
}

function downloadBlob(file: File) {
  const href = URL.createObjectURL(file);
  const a = document.createElement("a");
  a.href = href;
  a.download = file.name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 10_000);
}

export type SaveResult = "shared" | "downloaded" | "manual" | "failed";

/** Save one or many photos to the user's device.
 *  "shared"    — native share sheet opened (phone → Save to Photos)
 *  "downloaded"— saved via browser download (desktop, or share unsupported here)
 *  "manual"    — no automatic path works on this browser; caller should tell
 *                the user to long-press a photo and choose Save Image
 *  "failed"    — couldn't even fetch the photos
 */
export async function savePhotos(urls: string[]): Promise<SaveResult> {
  if (!urls.length) return "failed";

  const nav = navigator as Navigator & {
    canShare?: (d: any) => boolean;
    share?: (d: any) => Promise<void>;
  };

  // Fetch in PARALLEL — must stay fast enough that share() below is still
  // within the tap's activation window on iOS Safari.
  const files = await fetchFiles(urls);
  if (!files.length) return "failed";

  if (nav.share && nav.canShare?.({ files })) {
    try {
      await nav.share({ files, title: "SnapAI album" });
      return "shared";
    } catch (e: any) {
      if (e?.name === "AbortError") return "shared";   // user closed the sheet — fine
      // Any other failure (including the activation-expired case) falls
      // through to the platform-appropriate fallback below.
    }
  }

  if (isMobile()) {
    // `<a download>` does not save to the camera roll on mobile browsers —
    // don't claim success. The photos are already visible as normal <img>
    // tags in the album; long-press works natively, no JS needed.
    return "manual";
  }

  for (let i = 0; i < files.length; i++) {
    setTimeout(() => downloadBlob(files[i]), i * 350);
  }
  return "downloaded";
}
