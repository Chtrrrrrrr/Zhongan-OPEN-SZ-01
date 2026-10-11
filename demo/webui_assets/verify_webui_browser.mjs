/**
 * WebUI 浏览器侧验收（真实 Chrome + CDP，无需 npm 依赖，只用 Node 内置 WebSocket/fetch）。
 *
 * 为什么需要它：`verify_webui.py` 只能证明 HTTP 层正确（首页 200、锚点存在、端点可用）；
 * 页面里的 JS 是否真的把**真实数据**渲染出来、有没有横向溢出、样本卡是否被裁切、
 * 点击卡片能否载入完整报告、图片/视频检测能否跑通降级，只有真实浏览器能证明。
 *
 * 前置：先 `python api/server.py --port 8080`（真实模型 + 预热完成）。
 * 用法：node demo/webui_assets/verify_webui_browser.mjs [--chrome <path>] [--base-url http://127.0.0.1:8080]
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));   // <repo>/demo/webui_assets
const DEFAULT_SHOTS = path.resolve(HERE, "..", "..", "..", ".tmp", "webui-shots");  // <ws>/.tmp（仓库外，避免污染 git 工作区）

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf("--" + name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const BASE = arg("base-url", "http://127.0.0.1:8080");
const PORT = Number(arg("port", "9340"));
const CHROME = arg("chrome", [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium"
].find((p) => { try { return fs.existsSync(p); } catch (e) { return false; } }) || "google-chrome");
const SHOT_DIR = arg("shot-dir", DEFAULT_SHOTS);

const results = [];
const ck = (name, ok, detail = "") => { results.push([name, !!ok, detail]); };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${path.join(SHOT_DIR, "profile")}`,
  "--window-size=1680,1200", BASE + "/"
], { stdio: "ignore" });

async function main() {
  let target = null;
  for (let i = 0; i < 40 && !target; i++) {
    await wait(500);
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      target = list.find((t) => t.type === "page" && t.url.startsWith(BASE));
    } catch (e) { /* 等 Chrome 起来 */ }
  }
  if (!target) throw new Error(`找不到 Chrome 调试目标（Chrome 可用? ${CHROME}）`);

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
  const send = (method, params = {}) => new Promise((res) => { const mid = ++id; pending.set(mid, res); ws.send(JSON.stringify({ id: mid, method, params })); });
  const evaluate = async (expr) => {
    const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.result && r.result.exceptionDetails) throw new Error("eval: " + JSON.stringify(r.result.exceptionDetails).slice(0, 300));
    return r.result && r.result.result ? r.result.result.value : null;
  };
  const shot = async (file, clip) => {
    fs.mkdirSync(SHOT_DIR, { recursive: true });
    const r = await send("Page.captureScreenshot", clip
      ? { format: "png", clip: { ...clip, scale: 1 }, captureBeyondViewport: true }
      : { format: "png", captureBeyondViewport: true });
    fs.writeFileSync(file, Buffer.from(r.result.data, "base64"));
    return fs.statSync(file).size;
  };
  await send("Page.enable"); await send("Runtime.enable");
  await wait(8000);   // 等 boot() 的 4 个 fetch 与渲染完成

  // ---------- 1. 布局与内容度量 ----------
  const d = await evaluate(`(() => {
    const q = (s) => document.querySelector(s);
    const cards = [...document.querySelectorAll('#samplesGrid .sc')];
    const secs = {};
    for (const s of ['matrix','live','samples','dashboard','compliance']) {
      const e = q('#' + s); const b = e.getBoundingClientRect();
      secs[s] = { top: Math.round(b.top + window.scrollY), h: Math.round(b.height) };
    }
    return {
      docW: document.documentElement.scrollWidth, docH: document.documentElement.scrollHeight,
      viewW: window.innerWidth, horizontal: document.documentElement.scrollWidth > window.innerWidth + 1,
      cards: cards.length,
      fp: cards.filter(c => c.classList.contains('fp')).length,
      fn: cards.filter(c => c.classList.contains('fn')).length,
      clippedMeta: cards.filter(c => { const m = c.querySelector('.meta'); return m && m.scrollWidth > m.clientWidth + 1; }).length,
      sections: secs,
      status: q('#stHealth').textContent,
      thresholdChip: q('#stThr').textContent,
      kpis: [...document.querySelectorAll('#kpiBox .kpi')].map(k => k.querySelector('b').textContent),
      evidenceCards: document.querySelectorAll('#evidenceBox .card').length,
      evidenceMissing: document.querySelectorAll('#evidenceBox .note.warn').length,
      confusion: [...document.querySelectorAll('#confBox .cm div')].map(x => x.textContent.trim()).join('|'),
      leaked: (document.body.innerText.match(/undefined|NaN|\\[object Object\\]|&quot;|&lt;br&gt;/g) || []),
      cmds: document.querySelectorAll('#cmds .mono').length
    };
  })()`);
  ck("页面无横向溢出（scrollWidth == innerWidth）", !d.horizontal, `docW=${d.docW} viewW=${d.viewW} docH=${d.docH}`);
  ck("5 个板块锚点均有实际高度", Object.values(d.sections).every((s) => s.h > 100), JSON.stringify(d.sections));
  ck("样本库渲染 119 张卡片", d.cards === 119, "n=" + d.cards);
  ck("误报 2 / 漏报 3 着色正确", d.fp === 2 && d.fn === 3, `fp=${d.fp} fn=${d.fn}`);
  ck("样本卡片文本无裁切", d.clippedMeta === 0, "clipped=" + d.clippedMeta);
  ck("顶栏由真实 health 填充", /服务状态：(ok|degraded)/.test(d.status), d.status);
  ck("阈值来自真实 version", /low 0\.0364/.test(d.thresholdChip), d.thresholdChip);
  ck("指标看板 6 个 KPI 渲染（含 0.958/0.0317/0.9464）",
    d.kpis.length === 6 && d.kpis.some((k) => k.includes("0.958")) && d.kpis.some((k) => k.includes("0.0317")),
    d.kpis.join(" | "));
  ck("消融/指标证据卡片 ≥ 5 张", d.evidenceCards >= 5, "cards=" + d.evidenceCards);
  ck("混淆矩阵布局正确（真实 AI → TP 53 / FN 3）", /真实 AI\|TP 53\|FN 3/.test(d.confusion), d.confusion);
  ck("页面无 undefined/NaN/未转义实体泄漏", d.leaked.length === 0, JSON.stringify(d.leaked.slice(0, 5)));
  ck("复现命令块已渲染", d.cmds >= 5, "cmds=" + d.cmds);

  const full = await shot(path.join(SHOT_DIR, "full.png"));
  ck("全页截图可生成", full > 100000, full + " bytes");

  // ---------- 2. 样本库点击 → 完整报告弹窗 ----------
  await evaluate("openSample(57)");
  await wait(3000);
  const m = await evaluate(`(() => {
    const b = document.querySelector('#modalBody');
    const box = document.querySelector('#sampleFull');
    const t = b.innerText + ' ' + (box ? box.innerText : '');
    return { shown: document.querySelector('#modal').classList.contains('show'),
      fullLoaded: t.includes('已载入落盘完整报告'),
      hasL3: t.includes('伪造区域定位'), hasAttention: t.includes('关注区域解释'),
      hasClues: t.includes('L1 取证线索'), hasLatency: t.includes('逐环节耗时'),
      hasHash: t.includes('artifact_sha256'), heatmaps: [...b.querySelectorAll('img.hm')].filter(i => i.naturalWidth > 0).length };
  })()`);
  ck("点击样本卡片打开详情弹窗", m.shown);
  ck("载入该样本落盘的完整报告", m.fullLoaded);
  ck("完整报告含 L3 定位图与关注区域解释（术语区分）", m.hasL3 && m.hasAttention);
  ck("完整报告含取证线索/逐环节耗时/版本哈希", m.hasClues && m.hasLatency && m.hasHash);
  ck("热力图实际解码成功（naturalWidth>0）", m.heatmaps >= 1, "loaded=" + m.heatmaps);
  const mr = await evaluate("(() => { const r = document.querySelector('#modalBody').closest('.sheet').getBoundingClientRect(); return {x:0,y:Math.round(r.top+window.scrollY),width:Math.round(r.width)+80,height:Math.min(Math.round(r.height),2200)}; })()");
  await shot(path.join(SHOT_DIR, "sample_modal.png"), mr);
  await evaluate("closeModal()");

  // ---------- 3. 实时检测：图片（canvas 造真实 PNG → 原始字节上传） ----------
  await evaluate(`(async () => {
    const c = document.createElement('canvas'); c.width = 900; c.height = 620;
    const g = c.getContext('2d');
    g.fillStyle = '#f3f0e8'; g.fillRect(0, 0, 900, 620);
    g.strokeStyle = '#8a1b1b'; g.lineWidth = 3; g.strokeRect(30, 30, 840, 560);
    g.fillStyle = '#1a237e'; g.font = '28px sans-serif'; g.fillText('browser self-check', 70, 110);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    const dt = new DataTransfer(); dt.items.add(new File([blob], 'selfcheck.png', { type: 'image/png' }));
    const inp = document.querySelector('#file'); inp.files = dt.files;
    inp.dispatchEvent(new Event('change'));
    document.querySelector('#go').click();
    return true;
  })()`);
  let live = null;
  for (let i = 0; i < 60; i++) {
    await wait(1000);
    live = await evaluate(`(() => { const o = document.querySelector('#out');
      return { spinning: !!document.querySelector('#spin').textContent.trim(),
        badge: !!o.querySelector('.badge'), hasClues: o.innerText.includes('L1 取证线索'),
        hasLoc: o.innerText.includes('伪造区域定位'), heatmaps: [...o.querySelectorAll('img.hm')].filter(i=>i.naturalWidth>0).length }; })()`);
    if (!live.spinning && live.badge) break;
  }
  ck("图片实时检测在页面上出结论徽标", live.badge);
  ck("图片检测结果含取证线索与 L3 定位区", live.hasClues && live.hasLoc);
  ck("L3 定位/归因热力图解码成功", live.heatmaps >= 1, "loaded=" + live.heatmaps);

  // ---------- 4. 超时降级 ----------
  await evaluate("document.querySelector('#timeoutSel').value='1'; document.querySelector('#go').click();");
  let deg = null;
  for (let i = 0; i < 20; i++) {
    await wait(800);
    deg = await evaluate("(() => ({ degraded: !!document.querySelector('#out .note.warn'), text: document.querySelector('#out').innerText.slice(0,120) }))()");
    if (deg.degraded) break;
  }
  ck("超时降级在页面上提示而非报错", deg.degraded && /降级返回/.test(deg.text), (deg.text || "").replace(/\n+/g, " "));

  const fails = results.filter((r) => !r[1]);
  for (const [n, ok, dt] of results) console.log("[%s] %s%s", ok ? "PASS" : "FAIL", n, dt ? "  — " + dt : "");
  console.log("\n===== 浏览器侧验收：%d/%d 通过（截图见 %s）=====", results.length - fails.length, results.length, SHOT_DIR);
  ws.close(); chrome.kill();
  process.exit(fails.length ? 1 : 0);
}
main().catch((e) => { console.error("FAILED:", e.message); try { chrome.kill(); } catch (x) {} process.exit(1); });
