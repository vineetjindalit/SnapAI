// Dependency-free ZIP (store / no compression) + universal save helper.
// Enough to bundle the curated album JPEGs into ONE downloadable .zip in any
// browser — no jszip/fflate needed. Chrome/Edge get a native "save as" dialog
// (a real file explorer); every other browser falls back to a normal download.

function crc32(buf: Uint8Array): number {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (~c) >>> 0;
}

const u16 = (n: number) => new Uint8Array([n & 255, (n >>> 8) & 255]);
const u32 = (n: number) =>
  new Uint8Array([n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]);

function concat(parts: Uint8Array[]): Uint8Array {
  let len = 0; for (const p of parts) len += p.length;
  const out = new Uint8Array(len);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export interface ZipFile { name: string; data: Uint8Array; }

/** Build a valid (store-method) .zip from in-memory files. */
export function makeZip(files: ZipFile[]): Blob {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const f of files) {
    const name = enc.encode(f.name);
    const crc  = crc32(f.data);
    const size = f.data.length;
    const lfh = concat([
      u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(size), u32(size), u16(name.length), u16(0),
      name, f.data,
    ]);
    chunks.push(lfh);
    central.push(concat([
      u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(size), u32(size), u16(name.length), u16(0), u16(0),
      u16(0), u16(0), u32(0), u32(offset), name,
    ]));
    offset += lfh.length;
  }

  const centralStart = offset;
  let centralSize = 0;
  for (const c of central) { chunks.push(c); centralSize += c.length; }
  chunks.push(concat([
    u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
    u32(centralSize), u32(centralStart), u16(0),
  ]));
  return new Blob(chunks as BlobPart[], { type: "application/zip" });
}

/** Save a Blob: native "save as" file picker where supported, else download. */
export async function saveBlob(blob: Blob, suggestedName: string) {
  const w = window as any;
  if (w.showSaveFilePicker) {
    try {
      const handle = await w.showSaveFilePicker({
        suggestedName,
        types: [{ description: "Zip archive",
                  accept: { "application/zip": [".zip"] } }],
      });
      const ws = await handle.createWritable();
      await ws.write(blob);
      await ws.close();
      return;
    } catch (e: any) {
      if (e?.name === "AbortError") return;   // user cancelled the dialog
      // any other error → fall through to a normal download
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/** Fetch each album photo, bundle them into one .zip and save it. */
export async function downloadAlbumZip(
  photos: Array<{ url?: string | null; moment?: string }>,
  eventName: string,
): Promise<number> {
  const files: ZipFile[] = [];
  let i = 1;
  for (const p of photos) {
    if (!p?.url) continue;
    try {
      const res = await fetch(p.url);
      const data = new Uint8Array(await res.arrayBuffer());
      const moment = String(p.moment || "moment").replace(/[^\w-]+/g, "_");
      files.push({ name: `${String(i).padStart(2, "0")}_${moment}.jpg`, data });
      i++;
    } catch { /* skip a photo that fails to fetch */ }
  }
  if (!files.length) return 0;
  const safe = (eventName || "album").replace(/[^\w-]+/g, "_");
  await saveBlob(makeZip(files), `${safe}.zip`);
  return files.length;
}
