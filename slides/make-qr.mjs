/**
 * make-qr.mjs — genereert een echte, telefoon-scanbare QR-code (PNG) voor de live CSD-simulator.
 *
 * Waarom een script i.p.v. een handgetekende SVG-matrix: de vorige QR in slides.md was met de hand
 * opgebouwd uit losse modulen. Daardoor klopte de foutcorrectie/maskerkeuze niet en kon een
 * telefoon hem niet decoderen. Hier gebruiken we de zero-dependency encoder `uqr` (zelfde algoritme
 * als Nayuki's QR-Code-generator) en schrijven we de matrix zelf als 8-bit grijswaarden-PNG
 * (zlib zit in Node, dus geen extra dependencies).
 *
 * Gebruik:  cd slides && node make-qr.mjs
 * Output:   public/qr_csd_simulator.png  (+ samenvatting op stdout)
 */
import { deflateSync } from 'node:zlib'
import { writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { encode } from 'uqr'

const TARGET_URL =
  'https://raw.githack.com/MeesterDavid/Levertoxiciteit/main/cortical_spreading_depression_simulator.html'

const SCALE = 10 // pixels per QR-module in de PNG (430-490 px totaal)
const BORDER = 4 // quiet zone in modules (QR-spec schrijft minimaal 4 voor)
const ECC = 'M' // 15% foutcorrectie: beste balans voor projectie/foto van een scherm

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../public/qr_csd_simulator.png')

/* ---------- minimale PNG-encoder (8-bit grijswaarden, filter 0) ---------- */
const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

function crc32(buf) {
  let c = ~0
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8)
  return (~c) >>> 0
}

function pngChunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length, 0)
  const typeBuf = Buffer.from(type, 'ascii')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0)
  return Buffer.concat([len, typeBuf, data, crc])
}

function encodeGrayPng(width, height, rows) {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 0 // color type 0 = grijswaarden
  ihdr[10] = 0 // deflate
  ihdr[11] = 0 // adaptive filtering
  ihdr[12] = 0 // no interlace
  return Buffer.concat([
    sig,
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(rows, { level: 9 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
}

/* ---------- QR genereren ---------- */
const { data, size, version, maskPattern } = encode(TARGET_URL, {
  ecc: ECC,
  border: BORDER,
  boostEcc: false,
})

const px = size * SCALE
const rows = Buffer.alloc((px + 1) * px, 0xff)
for (let y = 0; y < px; y++) {
  const moduleRow = data[Math.floor(y / SCALE)]
  const lineStart = y * (px + 1)
  rows[lineStart] = 0 // filter byte
  for (let x = 0; x < px; x++) {
    rows[lineStart + 1 + x] = moduleRow[Math.floor(x / SCALE)] ? 0x00 : 0xff
  }
}

writeFileSync(OUT, encodeGrayPng(px, px, rows))

console.log(`QR-URL        : ${TARGET_URL} (${TARGET_URL.length} tekens)`)
console.log(`QR-versie     : ${version} -> ${size - 2 * BORDER}x${size - 2 * BORDER} modules (+${BORDER} quiet zone)`)
console.log(`ECC / masker  : ${ECC} / ${maskPattern}`)
console.log(`PNG           : ${px}x${px} px (${SCALE} px per module) -> ${OUT}`)
