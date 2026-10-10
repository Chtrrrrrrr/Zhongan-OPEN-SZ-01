#!/usr/bin/env python3
"""OPEN-SZ-01 demo service - zero external web dependencies (stdlib http.server).

Endpoints:
  GET  /                -> single-page demo UI (upload image / video, view verdict)
  POST /api/detect      -> raw file bytes in body, filename via ?name= ; returns JSON
  POST /api/detect_video-> same, for video files
  GET  /api/health      -> which evidence producers are loaded
  GET  /reports/<f>     -> serve pre-generated sample reports

Run:
  python demo/app.py --port 7860
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>OPEN-SZ-01 · AI生成材料反欺诈检测</title>
<style>
:root{--bg:#0f1420;--card:#182031;--line:#2a3550;--fg:#e8edf7;--mut:#93a1bd;
      --ok:#2e7d32;--warn:#f9a825;--bad:#c62828;--acc:#4c8dff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:'Microsoft YaHei',system-ui,sans-serif}
header{padding:20px 28px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px}
h1{font-size:19px;margin:0}
.sub{color:var(--mut);font-size:13px}
main{display:grid;grid-template-columns:minmax(320px,1fr) minmax(360px,1.1fr);gap:22px;padding:22px 28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
#drop{border:2px dashed var(--line);border-radius:10px;padding:34px 16px;text-align:center;
      color:var(--mut);cursor:pointer;transition:.15s}
#drop.hot{border-color:var(--acc);color:var(--fg);background:#1d2740}
button{background:var(--acc);border:0;color:#fff;padding:10px 18px;border-radius:7px;
       font-size:14px;cursor:pointer;margin-top:12px}
button:disabled{opacity:.5;cursor:default}
img.prev{max-width:100%;border-radius:8px;margin-top:12px;display:none}
.badge{display:inline-block;padding:8px 22px;border-radius:7px;font-size:19px;font-weight:700;color:#fff}
.normal{background:var(--ok)} .review{background:var(--warn)} .suspect{background:var(--bad)}
table{width:100%;border-collapse:collapse;margin-top:8px}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;font-size:13px;text-align:left}
th{color:var(--mut);font-weight:500}
.kv{color:var(--mut);font-size:13px;line-height:1.9}
.hm{max-width:100%;border-radius:8px;margin-top:8px}
.spin{color:var(--mut);font-size:13px;margin-top:10px}
code{background:#0c111b;padding:2px 6px;border-radius:4px;color:#9ecbff}
</style></head><body>
<header><h1>AI 生成材料反欺诈风控检测</h1>
<span class="sub">OPEN-SZ-01 · 众安 1024 AI Coding 挑战赛 · L0→L1→L2→L3→融合→三档决策</span></header>
<main>
 <div class="card">
  <div id="drop">拖拽图片 / 视频到此处，或点击选择文件<br><span class="sub">支持 png / jpg / mp4 / mov</span></div>
  <input id="file" type="file" accept="image/*,video/*" style="display:none">
  <img id="prev" class="prev">
  <div><button id="go" disabled>开始检测</button></div>
  <div class="spin" id="spin"></div>
 </div>
 <div class="card" id="out"><div class="kv">等待输入…</div></div>
</main>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),
      prev=document.getElementById('prev'),go=document.getElementById('go'),
      out=document.getElementById('out'),spin=document.getElementById('spin');
let picked=null,isVideo=false;
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');take(e.dataTransfer.files[0])};
file.onchange=()=>take(file.files[0]);
function take(f){if(!f)return;picked=f;isVideo=f.type.startsWith('video');
  drop.textContent=f.name;go.disabled=false;
  if(!isVideo){prev.src=URL.createObjectURL(f);prev.style.display='block'}}
go.onclick=async()=>{
  if(!picked)return;go.disabled=true;spin.textContent='推理中…（首次调用含模型预热）';
  const t0=performance.now();
  const r=await fetch('/api/detect'+(isVideo?'_video':'')+'?name='+encodeURIComponent(picked.name),
                      {method:'POST',body:picked});
  const j=await r.json();spin.textContent='';
  render(j,(performance.now()-t0)/1000);go.disabled=false;
};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(j,wall){
  if(j.error){out.innerHTML='<div class="kv" style="color:#ff9a9a">错误：'+esc(j.error)+'</div>';return}
  const cls={normal:'normal',review:'review',suspect:'suspect'}[j.verdict]||'review';
  let h='<div class="badge '+cls+'">'+esc(j.verdict_cn||j.verdict)+'</div>';
  h+='<p class="kv">校准后置信度 <b>'+Number(j.confidence).toFixed(3)+'</b> ｜ 融合分数 '
     +Number(j.fused_score).toFixed(3)+' ｜ 模型分歧度 '+Number(j.disagreement).toFixed(3)
     +' ｜ 前端往返 '+wall.toFixed(2)+'s</p>';
  h+='<h3 style="font-size:15px;margin:14px 0 4px">模型证据</h3><table><tr><th>模型/证据</th><th>校准分数</th></tr>';
  for(const k in (j.model_scores||{})) h+='<tr><td>'+esc(k)+'</td><td>'+Number(j.model_scores[k]).toFixed(4)+'</td></tr>';
  h+='</table>';
  h+='<h3 style="font-size:15px;margin:14px 0 4px">L1 取证线索</h3><table><tr><th>线索</th><th>状态</th><th>说明</th></tr>';
  for(const c of (j.forensic_clues||[])) h+='<tr><td>'+esc(c.name)+'</td><td>'+esc(c.status)+'</td><td>'+esc(c.detail)+'</td></tr>';
  h+='</table>';
  if(j.localization&&j.localization.heatmap_png_b64){
    h+='<h3 style="font-size:15px;margin:14px 0 4px">伪造区域定位（L3，零权重取证图）</h3>'
      +'<img class="hm" src="data:image/png;base64,'+j.localization.heatmap_png_b64+'">';
  }
  if(j.localization&&j.localization.suspicious_segments&&j.localization.suspicious_segments.length)
    h+='<p class="kv">可疑片段(秒)：'+JSON.stringify(j.localization.suspicious_segments)+'</p>';
  h+='<h3 style="font-size:15px;margin:14px 0 4px">不确定性与局限</h3><ul class="kv">'
     +((j.uncertainty||[]).map(u=>'<li>'+esc(u)+'</li>').join('')||'<li>无</li>')+'</ul>';
  h+='<h3 style="font-size:15px;margin:14px 0 4px">耗时 / 可追溯</h3><table>';
  for(const k in (j.latency_ms||{})) h+='<tr><td>'+esc(k)+'</td><td>'+Number(j.latency_ms[k]).toFixed(0)+' ms</td></tr>';
  h+='</table><p class="kv">输入哈希 <code>'+esc(String(j.input_hash).slice(0,32))+'…</code></p>';
  out.innerHTML=h;
}
</script></body></html>"""


def build_detector(args):
    from detector import AIGCDetector, PipelineConfig

    def resolve_clip(base: str):
        p = Path(base)
        if p.is_dir() and not (p / "config.json").exists():
            snaps = [d for d in p.iterdir() if d.is_dir()]
            if snaps:
                return str(snaps[0])
        return str(p) if p.exists() else None

    def opt(p):
        return p if Path(p).exists() else None

    cfg = PipelineConfig(
        clip_dir=resolve_clip(args.clip_dir),
        clip_probe=opt(args.clip_probe),
        l1_lr_model=opt(args.l1_model),
        forensic_lr=opt(args.forensic_lr),
        clip_domain_lr=opt(args.clip_domain_lr),
        combined_lr=opt(args.combined_lr),
        fusion_json=opt(args.fusion),
        npr_weights=opt(args.npr),
        device=args.device,
    )
    return AIGCDetector(cfg)


class Handler(BaseHTTPRequestHandler):
    detector = None
    root = None

    def log_message(self, fmt, *a):  # quieter console
        sys.stderr.write("[demo] " + (fmt % a) + "\n")

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/health":
            d = self.detector
            self._send(200, json.dumps({
                "clip": d.clip is not None,
                "clip_probe": d._clip_head is not None,
                "l1_model": d.l1_pack is not None,
                "npr": d.npr is not None,
                "aeroblade": d.blade is not None,
                "device": d.device,
            }).encode("utf-8"))
        elif parsed.path.startswith("/reports/"):
            name = Path(parsed.path).name
            f = (self.root / "reports" / name)
            if f.is_file():
                ctype = "text/html; charset=utf-8" if f.suffix == ".html" else "application/json"
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, b'{"error":"not found"}')
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/api/detect", "/api/detect_video"):
            self._send(404, b'{"error":"not found"}')
            return
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""
        if not data:
            self._send(400, b'{"error":"empty body"}')
            return
        q = urllib.parse.parse_qs(parsed.query)
        name = (q.get("name") or ["upload.bin"])[0]
        suffix = Path(name).suffix or (".mp4" if parsed.path.endswith("video") else ".png")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(data)
                tmp = tf.name
            t0 = time.perf_counter()
            if parsed.path.endswith("_video"):
                rep = self.detector.detect_video(tmp)
            else:
                rep, _ = self.detector.detect_image(tmp)
            rep.latency_ms["server_total"] = round((time.perf_counter() - t0) * 1000, 1)
            self._send(200, rep.to_json().encode("utf-8"))
        except Exception as e:  # never crash the demo
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"},
                                       ensure_ascii=False).encode("utf-8"))
        finally:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--clip-dir", default="models/hf_cache/models--openai--clip-vit-large-patch14/snapshots")
    ap.add_argument("--clip-probe", default="models/clip_probe_norm.pt")
    ap.add_argument("--l1-model", default="models/l1_lr.joblib")
    ap.add_argument("--forensic-lr", default="models/forensic_lr.joblib")
    ap.add_argument("--clip-domain-lr", default="models/clip_domain_lr.joblib")
    ap.add_argument("--combined-lr", default="models/combined_lr.joblib")
    ap.add_argument("--fusion", default="models/fusion.json")
    ap.add_argument("--npr", default="models/npr.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the startup warm-up call (cold first request ~14s)")
    args = ap.parse_args()

    Handler.detector = build_detector(args)
    Handler.root = ROOT

    if not args.no_warmup:
        # Model residency + warm-up keeps live requests on the warm path (manual 16.2):
        # the first inference otherwise pays ~14s of load + cudnn autotune.
        import time as _t
        warm = ROOT.parent / "data" / "images" / "001.png"
        t0 = _t.perf_counter()
        try:
            if warm.exists():
                Handler.detector.detect_image(str(warm))
                print(f"[demo] warm-up done in {_t.perf_counter() - t0:.1f}s")
            else:
                print("[demo] warm-up skipped (no data/images/001.png)")
        except Exception as e:
            print(f"[demo] warm-up failed (continuing): {e}")

    print(f"[demo] http://{args.host}:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
