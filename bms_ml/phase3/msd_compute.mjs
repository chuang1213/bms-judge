// Step 2/3 of the MinaCalc MSD build: read the row-mask blob, run MinaCalc, write JSONL.
//
// The WASM and its Emscripten glue come from the MMA reference project unchanged
// (ref_repo/osumania_map_analyser-main/ManiaMapAnalyser by Leo_Black/js/ett). Version
// 0.74.0 is not a taste choice: per MMA's own versions/index.js, non-4K keycounts are
// pinned to 0.74.0 because it is the first MinaCalc with a real n-key pipeline - our
// charts are 7-8 columns, so 0.72.x and older would reject them at the FFI gate.
//
// Usage: node msd_compute.mjs <charts.bin> <out.jsonl>
// Blob layout is documented in msd_prep.py.
import { readFileSync, writeFileSync, openSync, writeSync, closeSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..", "..");
const MMA = path.join(ROOT, "ref_repo", "osumania_map_analyser-main",
    "ManiaMapAnalyser by Leo_Black", "js", "ett");
const VERSION = "0.74.0";
const SKILLSETS = ["Overall", "Stream", "Jumpstream", "Handstream", "Stamina",
    "JackSpeed", "Chordjack", "Technical"];

const [binPath, outPath] = process.argv.slice(2);
if (!binPath || !outPath) {
    console.error("usage: node msd_compute.mjs <charts.bin> <out.jsonl>");
    process.exit(1);
}

const glueUrl = pathToFileURL(path.join(MMA, "versions", `minaclac-74.0.js`));
const { default: createMinaCalc } = await import(glueUrl.href);
const wasmBinary = new Uint8Array(
    readFileSync(path.join(MMA, "versions", `minaclac-74.0.wasm`)));
const module = await createMinaCalc({
    wasmBinary,
    locateFile: () => path.join(MMA, "versions", `minaclac-74.0.wasm`),
});
if (!module._minacalc_compute) {
    console.error("wasm glue does not export _minacalc_compute");
    process.exit(1);
}

const buf = readFileSync(binPath);
const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const fd = openSync(outPath, "w");
const decoder = new TextDecoder();

let off = 0;
let n_ok = 0, n_fail = 0;
const t0 = Date.now();
while (off < buf.byteLength) {
    const sha = decoder.decode(buf.subarray(off, off + 64));
    off += 64;
    const keycount = view.getUint32(off, true); off += 4;
    const nRows = view.getUint32(off, true); off += 4;
    const masks = new Uint32Array(buf.buffer, buf.byteOffset + off, nRows); off += 4 * nRows;
    const times = new Float32Array(buf.buffer, buf.byteOffset + off, nRows); off += 4 * nRows;

    const mPtr = module._malloc(nRows * 4);
    const tPtr = module._malloc(nRows * 4);
    const oPtr = module._malloc(SKILLSETS.length * 4);
    let values = null;
    try {
        module.HEAPU32.set(masks, mPtr >>> 2);
        module.HEAPF32.set(times, tPtr >>> 2);
        const ok = module._minacalc_compute(keycount, 1.0, 0.93, mPtr, tPtr, nRows, oPtr);
        if (ok) {
            const raw = module.HEAPF32.slice((oPtr >>> 2), (oPtr >>> 2) + SKILLSETS.length);
            values = Object.fromEntries(SKILLSETS.map((k, i) => [k, Number(raw[i]) || 0]));
            n_ok += 1;
        } else {
            n_fail += 1;
        }
    } catch (err) {
        n_fail += 1;
        writeSync(fd, JSON.stringify({ sha256: sha, error: String(err) }) + "\n");
        module._free(mPtr); module._free(tPtr); module._free(oPtr);
        continue;
    }
    module._free(mPtr); module._free(tPtr); module._free(oPtr);
    writeSync(fd, JSON.stringify({ sha256: sha, keycount, values }) + "\n");

    if (n_ok % 500 === 0) {
        const sec = (Date.now() - t0) / 1000;
        console.log(`${n_ok} done (${(n_ok / sec).toFixed(1)}/s)`);
    }
}
closeSync(fd);
console.log(`ok ${n_ok}, fail ${n_fail} -> ${outPath}`);
