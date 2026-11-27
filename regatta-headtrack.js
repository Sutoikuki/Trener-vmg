/* =====================================================================
   Regatta HeadTrack — śledzenie głowy i wzroku z kamery internetowej
   (MediaPipe Face Landmarker, przetwarzanie w 100% lokalne w przeglądarce)

   PODPIĘCIE DO SYMULATORA:
     <script src="regatta-headtrack.js"></script>
   jako PIERWSZY skrypt w <head> — PRZED Three.js. Moduł sam wyłapie
   każdy THREE.WebGLRenderer (przez hak __THREE_DEVTOOLS__) i doda
   obrót/przesunięcie głowy do kamery tylko na czas renderowania.
   Logika gry, sterowanie i kamera symulatora zostają bez zmian.

   Skróty:  F8 = wyzeruj (patrz prosto)   F9 = włącz/wyłącz
   API:     window.RegattaHead.{start, stop, center, toggle, attach,
            calibrateGaze, on, pose, gaze, config}
   Kamera wymaga HTTPS (GitHub Pages) albo localhost.
   ===================================================================== */
(function () {
  'use strict';
  if (window.RegattaHead) return;

  const MP_VER = '0.10.14';
  const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + MP_VER;
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const LS_KEY = 'regattaHead.v1';
  const DEG = Math.PI / 180;

  const defaults = {
    enabled: true, gainYaw: 2.2, gainPitch: 1.6, gainRoll: 0.6, gainMove: 1.0,
    deadzone: 1.5, maxYaw: 165, maxPitch: 70, smooth: 0.5,
    invYaw: false, invPitch: false, invRoll: false, useRoll: true, useMove: true,
    panelMin: false, gazeW: null
  };
  const cfg = Object.assign({}, defaults);
  try { Object.assign(cfg, JSON.parse(localStorage.getItem(LS_KEY) || '{}')); } catch (e) {}
  const save = () => { try { localStorage.setItem(LS_KEY, JSON.stringify(cfg)); } catch (e) {} };

  // --- stan ---------------------------------------------------------------
  const raw = { ok: false, yaw: 0, pitch: 0, roll: 0, x: 0, y: 0, size: 1, ex: 0, ey: 0 };
  const zero = { set: false, yaw: 0, pitch: 0, roll: 0, x: 0, y: 0, size: 1 };
  const pose = { tracking: false, yaw: 0, pitch: 0, roll: 0, x: 0, y: 0, z: 0 }; // rad, m
  const gaze = { ok: false, x: 0, y: 0 };                                        // px okna
  const listeners = { pose: [], gaze: [], status: [] };
  const emit = (ev, d) => { for (const f of listeners[ev]) { try { f(d); } catch (e) { console.error(e); } } };

  // --- filtr One-Euro (gładko w spoczynku, szybko w ruchu) ----------------
  function OneEuro(beta) { this.beta = beta; this.x = null; this.dx = 0; }
  OneEuro.prototype.f = function (v, dt, minCut) {
    if (this.x === null) { this.x = v; return v; }
    const a = c => 1 / (1 + 1 / (2 * Math.PI * c * dt));
    this.dx += a(1) * ((v - this.x) / dt - this.dx);
    this.x += a(minCut + this.beta * Math.abs(this.dx)) * (v - this.x);
    return this.x;
  };
  OneEuro.prototype.reset = function () { this.x = null; this.dx = 0; };
  const F = {
    yaw: new OneEuro(0.02), pitch: new OneEuro(0.02), roll: new OneEuro(0.02),
    x: new OneEuro(2), y: new OneEuro(2), size: new OneEuro(0.5),
    gx: new OneEuro(0.004), gy: new OneEuro(0.004)
  };

  // --- kamera + MediaPipe ---------------------------------------------------
  let video = null, landmarker = null, stream = null, running = false, loading = false;
  let lastVT = -1, lastT = 0, lost = 0, autoZero = 0;

  async function start() {
    if (running || loading) return;
    loading = true; ui.sync();
    try {
      if (!window.isSecureContext) throw new Error('kamera wymaga HTTPS lub localhost');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('przeglądarka nie udostępnia kamery');
      if (!landmarker) {
        status('Ładowanie modelu twarzy…');
        const vision = await import(MP_URL + '/vision_bundle.mjs');
        const files = await vision.FilesetResolver.forVisionTasks(MP_URL + '/wasm');
        const opts = d => ({
          baseOptions: { modelAssetPath: MODEL_URL, delegate: d },
          runningMode: 'VIDEO', numFaces: 1,
          outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false
        });
        try { landmarker = await vision.FaceLandmarker.createFromOptions(files, opts('GPU')); }
        catch (e) { landmarker = await vision.FaceLandmarker.createFromOptions(files, opts('CPU')); }
      }
      status('Uruchamianie kamery…');
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }
      });
      video.srcObject = stream;
      await video.play();
      running = true; zero.set = false; autoZero = 0; for (const k in F) F[k].reset();
      status('Patrz prosto w ekran — zeruję…');
      requestAnimationFrame(loop);
    } catch (e) {
      status('Błąd: ' + (e && e.message ? e.message : e));
    } finally { loading = false; ui.sync(); }
  }

  function stop() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; pose.tracking = false; gaze.ok = false;
    status('Zatrzymano'); ui.sync();
  }

  function loop() {
    if (!running) return;
    if (video.readyState >= 2 && video.currentTime !== lastVT) {
      lastVT = video.currentTime;
      const now = performance.now();
      let res = null;
      try { res = landmarker.detectForVideo(video, now); } catch (e) { console.warn(e); }
      process(res, now);
    }
    requestAnimationFrame(loop);
  }

  function process(res, now) {
    const dt = Math.min(0.2, Math.max(1 / 120, (now - lastT) / 1000 || 1 / 30));
    lastT = now;
    const lm = res && res.faceLandmarks && res.faceLandmarks[0];
    if (!lm) {
      raw.ok = false; lost += dt;
      if (lost > 0.3) {                      // brak twarzy: płynny powrót na wprost
        const k = Math.exp(-dt * 4);
        pose.yaw *= k; pose.pitch *= k; pose.roll *= k; pose.x *= k; pose.y *= k; pose.z *= k;
        gaze.ok = false;
      }
      ui.frame(null); emit('pose', pose);
      return;
    }
    lost = 0;
    const W = video.videoWidth || 640, H = video.videoHeight || 480;
    const P = i => [lm[i].x * W, lm[i].y * H, lm[i].z * W];
    const L = P(234), R = P(454), T = P(10), B = P(152);
    const rx = R[0] - L[0], ry = R[1] - L[1], rz = R[2] - L[2];
    const dx = B[0] - T[0], dy = B[1] - T[1], dz = B[2] - T[2];
    const faceW = Math.hypot(rx, ry, rz);

    // kąty głowy [°]: yaw + = obrót w swoje lewo, pitch + = w dół, roll + = przechył w lewo
    raw.yaw = Math.atan2(rz, Math.hypot(rx, ry)) / DEG;
    raw.pitch = Math.atan2(dz, Math.hypot(dx, dy)) / DEG;
    raw.roll = Math.atan2(ry, rx) / DEG;
    raw.x = ((L[0] + R[0]) / 2 - W / 2) / faceW;  // w szerokościach twarzy
    raw.y = ((L[1] + R[1]) / 2 - H / 2) / faceW;
    raw.size = faceW / W;

    // cechy oczu (tęczówka względem kącików i powiek) — do wzroku
    const eye = (iris, a, b, up, lo) => {
      const A = P(a), Bc = P(b), I = P(iris), U = P(up), D = P(lo);
      const ax = Bc[0] - A[0], ay = Bc[1] - A[1], w = Math.hypot(ax, ay) || 1;
      const mx = (A[0] + Bc[0]) / 2, my = (U[1] + D[1]) / 2;
      return [((I[0] - mx) * ax + (I[1] - my) * ay) / (w * w), (I[1] - my) / w];
    };
    if (lm.length > 473) {
      const e1 = eye(468, 33, 133, 159, 145), e2 = eye(473, 362, 263, 386, 374);
      raw.ex = (e1[0] + e2[0]) / 2; raw.ey = (e1[1] + e2[1]) / 2;
    }
    raw.ok = true;

    const mc = 0.4 + (1 - cfg.smooth) * 3;          // częstotliwość odcięcia filtra
    const s = {
      yaw: F.yaw.f(raw.yaw, dt, mc), pitch: F.pitch.f(raw.pitch, dt, mc), roll: F.roll.f(raw.roll, dt, mc),
      x: F.x.f(raw.x, dt, mc), y: F.y.f(raw.y, dt, mc), size: F.size.f(raw.size, dt, mc)
    };

    if (!zero.set) {                                 // auto-zero po ~0,7 s stabilnej twarzy
      autoZero += dt;
      if (autoZero > 0.7) { setZero(s); status('Śledzenie aktywne'); }
    }
    if (zero.set) {
      const c = (d, g, max) => {
        const a = Math.abs(d), dz2 = cfg.deadzone;
        if (a <= dz2) return 0;
        return Math.sign(d) * Math.min(max, (a - dz2) * g);
      };
      const sy = cfg.invYaw ? -1 : 1, sp = cfg.invPitch ? -1 : 1, sr = cfg.invRoll ? -1 : 1;
      pose.yaw = sy * c(s.yaw - zero.yaw, cfg.gainYaw, cfg.maxYaw) * DEG;
      pose.pitch = -sp * c(s.pitch - zero.pitch, cfg.gainPitch, cfg.maxPitch) * DEG;
      pose.roll = cfg.useRoll ? sr * c(s.roll - zero.roll, cfg.gainRoll, 40) * DEG : 0;
      if (cfg.useMove) {
        const g = cfg.gainMove;
        pose.x = -(s.x - zero.x) * 0.15 * g;           // 1 szer. twarzy ≈ 15 cm
        pose.y = -(s.y - zero.y) * 0.15 * g;
        pose.z = (zero.size / s.size - 1) * 0.6 * g;    // pochylenie do ekranu = kamera do przodu
      } else { pose.x = pose.y = pose.z = 0; }
      pose.tracking = true;
    }
    updateGaze(s, dt);
    ui.frame(lm);
    emit('pose', pose);
  }

  function setZero(s) {
    s = s || { yaw: F.yaw.x, pitch: F.pitch.x, roll: F.roll.x, x: F.x.x, y: F.y.x, size: F.size.x };
    if (s.yaw === null || s.yaw === undefined) return;
    Object.assign(zero, { set: true, yaw: s.yaw, pitch: s.pitch, roll: s.roll, x: s.x, y: s.y, size: s.size });
  }
  function center() { if (running) { setZero(); status('Wyzerowano'); } }
  function toggle() { cfg.enabled = !cfg.enabled; save(); ui.sync(); }

  // --- wzrok: regresja grzbietowa z kalibracji 9-punktowej ----------------
  const feat = () => [1, raw.ex, raw.ey, raw.yaw / 30, raw.pitch / 30, raw.x, raw.y];
  function solve(A, b) {                       // (AᵀA + λI) w = Aᵀb, eliminacja Gaussa
    const n = A[0].length, M = [];
    for (let i = 0; i < n; i++) {
      M.push(new Array(n + 1).fill(0));
      for (let j = 0; j < n; j++) for (const r of A) M[i][j] += r[i] * r[j];
      for (let k = 0; k < A.length; k++) M[i][n] += A[k][i] * b[k];
    }
    for (let i = 0; i < n; i++) M[i][i] += 1e-6 + M[i][i] * 1e-3;
    for (let i = 0; i < n; i++) {
      let p = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
      [M[i], M[p]] = [M[p], M[i]];
      for (let r = 0; r < n; r++) {
        if (r === i || !M[i][i]) continue;
        const f = M[r][i] / M[i][i];
        for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c];
      }
    }
    return M.map((row, i) => row[n] / (row[i] || 1));
  }
  function updateGaze(s, dt) {
    const w = cfg.gazeW;
    if (!w || !raw.ok) { gaze.ok = false; return; }
    const f = feat();
    let gx = 0, gy = 0;
    for (let i = 0; i < f.length; i++) { gx += f[i] * w[0][i]; gy += f[i] * w[1][i]; }
    gaze.x = F.gx.f(gx * innerWidth, dt, 0.6 + (1 - cfg.smooth));
    gaze.y = F.gy.f(gy * innerHeight, dt, 0.6 + (1 - cfg.smooth));
    gaze.ok = true;
    emit('gaze', gaze);
  }

  let calibrating = false;
  async function calibrateGaze() {
    if (calibrating) return null;
    if (!running) { await start(); if (!running) return null; }
    calibrating = true;
    const ov = document.createElement('div');
    ov.id = 'rht-cal';
    ov.innerHTML = '<div class="rht-cal-msg">Kalibracja wzroku: nie ruszaj głową, patrz w środek kropki.<br>Esc = przerwij</div><div class="rht-cal-dot"></div>';
    document.body.appendChild(ov);
    const dot = ov.querySelector('.rht-cal-dot'), msg = ov.querySelector('.rht-cal-msg');
    let abort = false;
    const onKey = e => { if (e.key === 'Escape') abort = true; };
    addEventListener('keydown', onKey);
    const wait = ms => new Promise(r => setTimeout(r, ms));
    const pts = [];
    for (const y of [0.1, 0.5, 0.9]) for (const x of [0.08, 0.5, 0.92]) pts.push([x, y]);
    pts.sort(() => Math.random() - 0.5);
    const A = [], bx = [], by = [];
    await wait(1500); msg.style.opacity = '0.35';
    for (const [x, y] of pts) {
      if (abort) break;
      dot.style.left = x * 100 + '%'; dot.style.top = y * 100 + '%';
      dot.classList.remove('go'); void dot.offsetWidth; dot.classList.add('go');
      await wait(800);
      const t0 = performance.now();
      while (performance.now() - t0 < 1000 && !abort) {
        if (raw.ok) { A.push(feat()); bx.push(x); by.push(y); }
        await wait(33);
      }
    }
    removeEventListener('keydown', onKey);
    ov.remove(); calibrating = false;
    if (abort || A.length < 60) { status(abort ? 'Kalibrację przerwano' : 'Za mało próbek — sprawdź oświetlenie'); return null; }
    const w = [solve(A, bx), solve(A, by)];
    let err = 0;
    for (let k = 0; k < A.length; k++) {
      let px = 0, py = 0;
      for (let i = 0; i < A[k].length; i++) { px += A[k][i] * w[0][i]; py += A[k][i] * w[1][i]; }
      err += Math.hypot((px - bx[k]) * innerWidth, (py - by[k]) * innerHeight);
    }
    err /= A.length;
    cfg.gazeW = w; save(); F.gx.reset(); F.gy.reset();
    status('Wzrok skalibrowany, średni błąd ≈ ' + Math.round(err) + ' px');
    return { errorPx: err };
  }

  // --- hak na Three.js: offset głowy tylko na czas render() ---------------
  const hooked = new Set();
  function attach(r) {
    if (!r || hooked.has(r) || typeof r.render !== 'function') return;
    hooked.add(r);
    const orig = r.render;
    let tq = null, te = null, tv = null, sq = null, sp = null;
    r.render = function (scene, cam) {
      const act = cfg.enabled && pose.tracking && cam && cam.isPerspectiveCamera &&
        !(cam.userData && cam.userData.noHeadTrack);
      if (!act) return orig.apply(this, arguments);
      if (!tq) { tq = cam.quaternion.clone(); te = cam.rotation.clone(); tv = cam.position.clone(); sq = tq.clone(); sp = tv.clone(); }
      sq.copy(cam.quaternion); sp.copy(cam.position);
      te.set(pose.pitch, pose.yaw, pose.roll, 'YXZ'); tq.setFromEuler(te);
      tv.set(pose.x, pose.y, pose.z).applyQuaternion(sq);
      cam.quaternion.multiply(tq); cam.position.add(tv); cam.updateMatrixWorld(true);
      try { return orig.apply(this, arguments); }
      finally { cam.quaternion.copy(sq); cam.position.copy(sp); cam.updateMatrixWorld(true); }
    };
    ui.sync();
  }
  (function installHook() {
    let dt = window.__THREE_DEVTOOLS__;
    if (!dt) { try { dt = new EventTarget(); window.__THREE_DEVTOOLS__ = dt; } catch (e) { return; } }
    dt.addEventListener('observe', e => { const o = e.detail; if (o && o.isWebGLRenderer) attach(o); });
  })();

  // --- panel --------------------------------------------------------------
  function status(t) { ui.status(t); emit('status', t); }
  const CSS = `
#rht{position:fixed;left:10px;bottom:10px;z-index:2147483000;width:232px;font:12px/1.35 system-ui,sans-serif;color:#d9ecf7;background:#0b1a26ee;border:1px solid #23435a;border-radius:12px;padding:8px;backdrop-filter:blur(6px);user-select:none;-webkit-user-select:none}
#rht.min{width:auto}#rht.min .rht-body{display:none}
#rht .rht-head{display:flex;align-items:center;gap:6px}
#rht .rht-dot{width:9px;height:9px;border-radius:50%;background:#56697a;flex:none}
#rht .rht-dot.on{background:#4fe08f;box-shadow:0 0 8px #4fe08f}#rht .rht-dot.lost{background:#ffb347}
#rht .rht-t{font-weight:700;letter-spacing:.04em;flex:1}
#rht button{font:inherit;color:#d9ecf7;background:#14303f;border:1px solid #2b5670;border-radius:7px;padding:5px 8px;min-height:30px;cursor:pointer}
#rht button:hover{background:#1b4157}#rht button.act{background:#1f6f8f;border-color:#3fd8ff}
#rht .rht-row{display:flex;gap:5px;margin-top:6px;flex-wrap:wrap}#rht .rht-row button{flex:1}
#rht .rht-cam{position:relative;margin-top:6px;height:96px;border-radius:8px;overflow:hidden;background:#061019}
#rht video,#rht canvas.rht-ov{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
#rht .rht-read{position:absolute;right:5px;bottom:4px;font:11px ui-monospace,monospace;text-shadow:0 1px 2px #000}
#rht .rht-st{margin-top:5px;color:#8fb3c9;min-height:16px}
#rht details{margin-top:6px}#rht summary{cursor:pointer;color:#8fb3c9}
#rht label{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-top:4px}
#rht input[type=range]{width:110px}
#rht-cal{position:fixed;inset:0;z-index:2147483001;background:#04101a;cursor:none}
#rht-cal .rht-cal-msg{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:#cfe6f5;font:18px/1.5 system-ui;text-align:center;transition:opacity .4s}
#rht-cal .rht-cal-dot{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;background:#ffb347;box-shadow:0 0 0 6px #ffb34733;left:-99px}
#rht-cal .rht-cal-dot::after{content:"";position:absolute;left:10px;top:10px;width:6px;height:6px;border-radius:50%;background:#04101a}
#rht-cal .rht-cal-dot.go{animation:rhtShrink 1.8s ease-out forwards}
@keyframes rhtShrink{0%{transform:scale(1.8)}45%{transform:scale(.8)}100%{transform:scale(.6)}}`;

  const sliders = [
    ['gainYaw', 'Obrót (yaw)', 0.5, 5, 0.1], ['gainPitch', 'Góra/dół', 0.5, 4, 0.1],
    ['gainRoll', 'Przechył', 0, 1.5, 0.05], ['gainMove', 'Wychylenie', 0, 3, 0.1],
    ['smooth', 'Wygładzanie', 0, 1, 0.05], ['deadzone', 'Martwa strefa °', 0, 8, 0.5]
  ];
  const checks = [['useRoll', 'Przechył głowy'], ['useMove', 'Ruch głowy (6DoF)'],
    ['invYaw', 'Odwróć obrót'], ['invPitch', 'Odwróć góra/dół'], ['invRoll', 'Odwróć przechył']];

  const ui = {
    el: null, ctx: null, st: '', read: null,
    build() {
      if (this.el) return;
      const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
      const el = document.createElement('div'); el.id = 'rht'; if (cfg.panelMin) el.classList.add('min');
      el.innerHTML = `
<div class="rht-head"><span class="rht-dot"></span><span class="rht-t">HEAD TRACK</span>
<button data-a="min" title="Zwiń/rozwiń">▾</button></div>
<div class="rht-body">
<div class="rht-row"><button data-a="run">▶ Start</button><button data-a="zero" title="F8">⦿ Zeruj</button><button data-a="en" title="F9">ON</button></div>
<div class="rht-cam"><video muted playsinline></video><canvas class="rht-ov" width="160" height="120"></canvas><div class="rht-read"></div></div>
<div class="rht-st"></div>
<details><summary>Ustawienia</summary>
${sliders.map(([k, l, a, b, s]) => `<label>${l}<input type="range" data-k="${k}" min="${a}" max="${b}" step="${s}"></label>`).join('')}
${checks.map(([k, l]) => `<label>${l}<input type="checkbox" data-k="${k}"></label>`).join('')}
<div class="rht-row"><button data-a="gaze">👁 Kalibruj wzrok</button><button data-a="reset">Domyślne</button></div>
</details></div>`;
      document.body.appendChild(el);
      this.el = el;
      video = el.querySelector('video');
      this.ctx = el.querySelector('canvas').getContext('2d');
      this.read = el.querySelector('.rht-read');
      el.addEventListener('click', e => {
        const a = e.target.closest('button') && e.target.closest('button').dataset.a;
        if (a === 'run') running ? stop() : start();
        else if (a === 'zero') center();
        else if (a === 'en') toggle();
        else if (a === 'gaze') calibrateGaze();
        else if (a === 'min') { cfg.panelMin = !cfg.panelMin; el.classList.toggle('min', cfg.panelMin); save(); }
        else if (a === 'reset') { const g = cfg.gazeW; Object.assign(cfg, defaults, { gazeW: g }); save(); this.sync(); }
      });
      el.addEventListener('input', e => {
        const k = e.target.dataset.k; if (!k) return;
        cfg[k] = e.target.type === 'checkbox' ? e.target.checked : parseFloat(e.target.value); save();
      });
      // klawisze nie trafiają do gry, gdy fokus jest w panelu
      el.addEventListener('keydown', e => e.stopPropagation());
      el.addEventListener('pointerdown', e => e.stopPropagation());
      el.addEventListener('touchstart', e => e.stopPropagation(), { passive: true });
      this.sync();
    },
    sync() {
      if (!this.el) return;
      const q = s => this.el.querySelector(s);
      q('[data-a=run]').textContent = loading ? '… ładuję' : running ? '■ Stop' : '▶ Start';
      const en = q('[data-a=en]'); en.textContent = cfg.enabled ? 'ON' : 'OFF'; en.classList.toggle('act', cfg.enabled);
      for (const i of this.el.querySelectorAll('[data-k]')) {
        if (i.type === 'checkbox') i.checked = !!cfg[i.dataset.k]; else i.value = cfg[i.dataset.k];
      }
      if (!running && !loading && !this.st) this.status(hooked.size ? 'Podpięto do sceny 3D. Kliknij Start.' : 'Kliknij Start, aby włączyć kamerę.');
    },
    status(t) { this.st = t; if (this.el) this.el.querySelector('.rht-st').textContent = t; },
    frame(lm) {
      if (!this.el) return;
      const dot = this.el.querySelector('.rht-dot');
      dot.classList.toggle('on', !!lm && cfg.enabled); dot.classList.toggle('lost', !lm && running);
      if (cfg.panelMin) return;
      const c = this.ctx; c.clearRect(0, 0, 160, 120);
      if (lm) {
        c.fillStyle = '#3fd8ff';
        for (let i = 0; i < lm.length; i += 6) c.fillRect(lm[i].x * 160, lm[i].y * 120, 1.5, 1.5);
        c.fillStyle = '#ffb347';
        for (const i of [468, 473]) if (lm[i]) c.fillRect(lm[i].x * 160 - 1.5, lm[i].y * 120 - 1.5, 3, 3);
      }
      this.read.textContent = lm
        ? `${Math.round(pose.yaw / DEG)}° ${Math.round(pose.pitch / DEG)}°`
        : 'brak twarzy';
    }
  };

  addEventListener('keydown', e => {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === 'F8') { e.preventDefault(); center(); }
    else if (e.key === 'F9') { e.preventDefault(); toggle(); }
  });

  if (document.body) ui.build(); else document.addEventListener('DOMContentLoaded', () => ui.build());

  window.RegattaHead = {
    start, stop, center, toggle, attach, calibrateGaze,
    on(ev, f) { if (listeners[ev]) listeners[ev].push(f); },
    pose, gaze, raw, config: cfg,
    get running() { return running; }
  };
})();
