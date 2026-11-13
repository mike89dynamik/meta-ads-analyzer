# v2 "kinetic collage" build for the 3.500 EUR reel. Generates compositions/s1..s6.html + index.html.
import random

O, K, W, R = "#E37739", "#15110E", "#F4F1EC", "#C8321E"
WHITE = "#FFFFFF"


def jag(seed, n=12, amp=1.6):
    """Deterministic torn-paper polygon (percent coords)."""
    r = random.Random(seed)
    pts = []
    for i in range(n + 1):  # top edge L->R
        pts.append((i / n * 100, r.uniform(0, amp)))
    for i in range(1, n + 1):  # right edge T->B
        pts.append((100 - r.uniform(0, amp), i / n * 100))
    for i in range(n - 1, -1, -1):  # bottom edge R->L
        pts.append((i / n * 100, 100 - r.uniform(0, amp)))
    for i in range(n - 1, 0, -1):  # left edge B->T
        pts.append((r.uniform(0, amp), i / n * 100))
    return "polygon(" + ", ".join(f"{x:.2f}% {y:.2f}%" for x, y in pts) + ")"


def card(cid, x, y, w, h, media, seed, extra=""):
    return f'''      <div id="{cid}" class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">
        <div class="paper" style="clip-path:{jag(seed)}"><div class="ph" style="clip-path:{jag(seed + 99, amp=0.8)}">{media}{extra}</div></div>
      </div>'''


def img(src):
    return f'<img src="assets/img/{src}" alt="" />'


def vid(vid_id, src, start, dur, rate=None):
    r = f' data-playback-rate="{rate}"' if rate else ""
    return f'<video id="{vid_id}" class="clip" src="assets/video/{src}" data-start="{start}" data-duration="{dur}"{r} muted playsinline></video>'


def fit(text, maxpx, width=980, k=0.74):
    return int(min(maxpx, width / (max(len(text), 1) * k)))


def tb(tid, html, top, size, color, plain_len=None):
    return f'      <div id="{tid}" class="tb" style="top:{top}px;font-size:{size}px;color:{color}"><span class="tbi">{html}</span></div>'


CSS = """
    #root { position: absolute; inset: 0; background: transparent; overflow: hidden; font-family: "Archivo Black", "Inter", sans-serif; }
    .bg { position: absolute; inset: 0; }
    .cam { position: absolute; inset: 0; perspective: 1400px; }
    .card { position: absolute; filter: drop-shadow(0 26px 28px rgba(0,0,0,0.38)); }
    .paper { position: absolute; inset: 0; background: #f7f3ea; }
    .ph { position: absolute; inset: 14px; overflow: hidden; }
    .ph img, .ph video { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: cover; }
    .tb { position: absolute; left: 30px; right: 30px; text-align: center; line-height: 0.92; letter-spacing: -0.02em; text-transform: uppercase; z-index: 20; }
    .tb em { font-style: normal; color: #E37739; }
    .tbi { display: inline-block; }
    .tag { position: absolute; left: 50%; bottom: 22px; transform: translateX(-50%); white-space: nowrap; padding: 10px 22px; border-radius: 14px; background: #15110E; color: #fff; font-size: 34px; }
"""

HELPERS = r"""
    function punch(sel, t, from) {
      tl.fromTo(sel + " .tbi", { opacity: 0, scale: from || 1.9 }, { opacity: 1, scale: 1, duration: 0.16, ease: "back.out(2.2)" }, t);
    }
    function cut(sel, t) { tl.set(sel + " .tbi", { opacity: 0 }, t); }
    function shake(t, amp) {
      const a = amp || 14;
      tl.fromTo(CAM, { x: 0, y: 0 }, { x: a, y: -a * 0.6, duration: 0.035, ease: "none", yoyo: true, repeat: 5 }, t);
      tl.set(CAM, { x: 0, y: 0 }, t + 0.22);
    }
    function bg(color, t) { tl.set(BG, { backgroundColor: color }, t); }
    function counter(sel, from, to, t, dur, fmt) {
      const el = document.querySelector(sel); const c = { v: from };
      tl.fromTo(c, { v: from }, { v: to, duration: dur, ease: "power2.out", onUpdate: () => { el.textContent = fmt(c.v); } }, t);
    }
    const eur = (v) => String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, ".") + " €";
"""


def scene(sid, dur, body, js, extra_css=""):
    return f"""<template>
  <style>{CSS}{extra_css}
  </style>
  <div id="root" data-composition-id="{sid}" data-width="1080" data-height="1350">
    <div id="{sid}-bg" class="bg"></div>
    <div id="{sid}-cam" class="cam" data-layout-allow-overflow data-layout-allow-overlap>
{body}
    </div>
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    const CAM = "#{sid}-cam", BG = "#{sid}-bg";{HELPERS}
{js}
    window.__timelines["{sid}"] = tl;
  </script>
</template>
"""


files = {}

# ============ S1 hook: 0.00 - 4.40 (abs) =====================================
body = "\n".join([
    card("a-cash", 250, 470, 580, 720, img("i1.jpg"), 11),
    card("a-hand", 250, 470, 580, 720, vid("a-v1", "v1.mp4", 1.25, 1.75), 12),
    card("a-contr", 200, 420, 680, 840, img("i7.jpg"), 13),
    tb("a-t1", "Investiresti", 170, fit("INVESTIRESTI", 150), K),
    tb("a-t2", '<span id="a-n1">3.500 €</span>', 120, 230, WHITE),
    tb("a-t3", "Per chiudere", 170, fit("PER CHIUDERE", 150), WHITE),
    tb("a-t4", '<span id="a-n2">100.000 €</span>', 120, 165, O),
    tb("a-t5", "di contratti?", 300, 88, K),
])
js = """
    bg("#F4F1EC", 0); bg("#15110E", 0.82); bg("#E37739", 2.35); bg("#F4F1EC", 2.95);
    // frame 0: word + cash card already landing
    tl.fromTo("#a-t1 .tbi", { opacity: 1, scale: 1.12 }, { opacity: 1, scale: 1, duration: 0.3, ease: "back.out(2)" }, 0);
    tl.fromTo("#a-cash", { y: -140, rotation: -14 }, { y: 0, rotation: -6, duration: 0.4, ease: "back.out(1.4)" }, 0);
    cut("#a-t1", 0.82);
    // 3.500
    punch("#a-t2", 0.82, 2.4); counter("#a-n1", 0, 3500, 0.82, 0.45, eur); shake(0.86, 16);
    tl.to("#a-cash", { scale: 1.06, duration: 0.08, yoyo: true, repeat: 1, ease: "power2.out" }, 0.82);
    tl.fromTo("#a-hand", { x: 1000, rotation: 18 }, { x: 0, rotation: 4, duration: 0.35, ease: "expo.out" }, 1.25);
    tl.to("#a-cash", { x: -1100, rotation: -24, duration: 0.3, ease: "power3.in" }, 1.22);
    cut("#a-t2", 2.35);
    // per chiudere
    punch("#a-t3", 2.35);
    tl.to("#a-hand", { y: 120, scale: 0.78, rotation: -8, duration: 0.3, ease: "power3.inOut" }, 2.35);
    cut("#a-t3", 2.95);
    // 100.000: the contracts slam down
    tl.to("#a-hand", { y: 1400, duration: 0.25, ease: "power3.in" }, 2.95);
    tl.fromTo("#a-contr", { y: -1700, rotation: 14, scale: 1.1 }, { y: 0, rotation: -3, scale: 1, duration: 0.26, ease: "power4.in" }, 2.92);
    shake(3.18, 20);
    punch("#a-t4", 2.95, 2.4); counter("#a-n2", 3500, 100000, 2.95, 0.4, eur);
    punch("#a-t5", 3.71, 1.5);
    tl.fromTo("#a-contr", { rotation: -3 }, { rotation: -1, duration: 1.0, ease: "none" }, 3.3);
"""
files["s1"] = scene("s1", 4.4, body, js)

# ============ S2: 4.28 - 10.18  (local = abs - 4.28) ==========================
body = "\n".join([
    card("b-show", 230, 380, 620, 900, vid("b-v2", "v2.mp4", 0, 3.4, 0.75), 21),
    card("b-pos", 540, 500, 480, 620, img("i3.jpg"), 22),
    card("b-contr", 110, 520, 540, 680, img("i7.jpg"), 23, '<span class="tag">100.000 €</span>'),
    card("b-cash", 650, 700, 330, 420, img("i1.jpg"), 24, '<span class="tag" style="font-size:28px">3.500 €</span>'),
    tb("b-t1", "È quanto pesa", 170, fit("E QUANTO PESA", 140), WHITE),
    tb("b-t2", "il marketing", 170, fit("IL MARKETING", 150), O),
    tb("b-t3", "sui serramentisti", 150, fit("SERRAMENTISTI", 140), WHITE),
    tb("b-t4", "che lavorano<br/>con noi", 140, 104, K),
    tb("b-t5", "al<br/>massimo", 430, 190, K),
    tb("b-t6", '≤<span id="b-n">3,5</span>%', 110, 250, WHITE),
    tb("b-t7", "del fatturato", 360, 84, K),
])
js = """
    bg("#15110E", 0); bg("#E37739", 1.17); bg("#F4F1EC", 3.10); bg("#E37739", 3.82);
    tl.fromTo("#b-show", { y: 1100, rotation: 10 }, { y: 0, rotation: 2, duration: 0.4, ease: "expo.out" }, 0);
    tl.fromTo("#b-show", { scale: 1 }, { scale: 1.04, duration: 0.75, ease: "none" }, 0.4);
    punch("#b-t1", 0.02); cut("#b-t1", 0.61);
    punch("#b-t2", 0.61); cut("#b-t2", 1.17); shake(0.63, 10);
    // serramentisti: showroom slides left, the installer arrives
    punch("#b-t3", 1.17);
    tl.to("#b-show", { x: -290, scale: 0.78, rotation: -7, duration: 0.3, ease: "power3.inOut" }, 1.17);
    tl.fromTo("#b-pos", { x: 900, rotation: 20 }, { x: 0, rotation: 6, duration: 0.32, ease: "expo.out" }, 1.2);
    cut("#b-t3", 1.92);
    punch("#b-t4", 1.92, 1.5); cut("#b-t4", 3.10);
    tl.to("#b-pos", { rotation: 2, x: -20, duration: 1.0, ease: "none" }, 1.95);
    // al massimo: everything clears, beat of white
    tl.to("#b-show", { x: -1300, rotation: -20, duration: 0.25, ease: "power3.in" }, 3.0);
    tl.to("#b-pos", { x: 1100, rotation: 20, duration: 0.25, ease: "power3.in" }, 3.0);
    punch("#b-t5", 3.10, 1.6); cut("#b-t5", 3.82);
    // 3,5%: the ratio as two pieces of paper
    punch("#b-t6", 3.82, 2.6); shake(3.86, 22);
    counter("#b-n", 0, 3.5, 3.82, 0.5, (v) => v.toFixed(1).replace(".", ","));
    tl.fromTo("#b-contr", { y: 1100, rotation: -16 }, { y: 0, rotation: -5, duration: 0.35, ease: "expo.out" }, 3.9);
    tl.fromTo("#b-cash", { y: 1100, rotation: 20 }, { y: 0, rotation: 8, duration: 0.35, ease: "expo.out" }, 4.05);
    punch("#b-t7", 4.94, 1.5);
    tl.to("#b-contr", { rotation: -3, duration: 1.4, ease: "none" }, 4.3);
"""
files["s2"] = scene("s2", 5.9, body, js)

# ============ S3: 10.10 - 13.35 (local = abs - 10.10) =========================
invoice = '''<div class="inv"><b>Fattura agenzia</b><span>Canone mensile</span><strong>2.000 €</strong></div>'''
body = "\n".join([
    f'''      <div id="c-invL" class="card half" style="left:180px;top:600px;width:720px;height:440px"><div class="paper" style="clip-path:polygon(0% 0%, 52% 0%, 47% 18%, 53% 34%, 46% 52%, 52% 70%, 47% 86%, 51% 100%, 0% 100%)">{invoice}</div></div>''',
    f'''      <div id="c-invR" class="card half" style="left:180px;top:600px;width:720px;height:440px"><div class="paper" style="clip-path:polygon(52% 0%, 100% 0%, 100% 100%, 51% 100%, 47% 86%, 52% 70%, 46% 52%, 53% 34%, 47% 18%)">{invoice}</div></div>''',
    card("c-pos", 230, 440, 620, 860, vid("c-v3", "v3.mp4", 1.5, 1.85), 31),
    tb("c-t1", "0 €", 120, 300, O),
    tb("c-t2", "canone mensile", 430, fit("CANONE MENSILE", 96), WHITE),
    tb("c-t3", "Paghi solo", 160, fit("PAGHI SOLO", 150), K),
    tb("c-t4", "appuntamenti", 160, fit("APPUNTAMENTI", 150), O),
    tb("c-t5", "qualificati", 160, fit("QUALIFICATI", 150), K),
])
css = """
    .inv { position: absolute; inset: 0; padding: 50px 60px; color: #15110E; font-family: "Inter", sans-serif; }
    .inv b { display: block; font-size: 34px; letter-spacing: 0.08em; text-transform: uppercase; color: #8a8580; }
    .inv span { display: block; margin-top: 40px; font-size: 52px; font-weight: 700; }
    .inv strong { display: block; margin-top: 18px; font-size: 120px; font-weight: 900; letter-spacing: -0.04em; color: #C8321E; }
"""
js = """
    bg("#15110E", 0); bg("#F4F1EC", 1.55);
    tl.fromTo("#c-invL, #c-invR", { y: 1000, rotation: 10 }, { y: 0, rotation: -3, duration: 0.22, ease: "expo.out" }, 0.0);
    punch("#c-t1", 0.11, 2.6); shake(0.15, 18);
    punch("#c-t2", 0.41, 1.5);
    // the fixed fee is torn in half
    tl.to("#c-invL", { x: -1000, y: 700, rotation: -40, duration: 0.5, ease: "power2.in" }, 0.6);
    tl.to("#c-invR", { x: 1000, y: 750, rotation: 44, duration: 0.5, ease: "power2.in" }, 0.6);
    cut("#c-t1", 1.55); cut("#c-t2", 1.55);
    // the installer: only qualified appointments
    tl.fromTo("#c-pos", { y: 1200, rotation: -12 }, { y: 0, rotation: -3, duration: 0.35, ease: "expo.out" }, 1.5);
    tl.to("#c-pos", { rotation: 1, duration: 1.4, ease: "none" }, 1.85);
    punch("#c-t3", 1.55); cut("#c-t3", 2.18);
    punch("#c-t4", 2.18); cut("#c-t4", 2.69); shake(2.2, 10);
    punch("#c-t5", 2.69);
"""
files["s3"] = scene("s3", 3.25, body, js, css)

# ============ S4: 13.35 - 17.30 (local = abs - 13.35) =========================
WIN = [(240, 316, 152, 226), (688, 316, 152, 226), (240, 812, 152, 221), (688, 812, 152, 221)]
sc = 732 / 1080
glows = "".join(
    f'<i id="d-w{i+1}" class="glow" style="left:{x*sc:.0f}px;top:{y*sc:.0f}px;width:{w*sc:.0f}px;height:{h*sc:.0f}px"></i>'
    for i, (x, y, w, h) in enumerate(WIN))
body = "\n".join([
    card("d-house", 160, 340, 760, 950, img("i4.jpg"), 41, glows),
    tb("d-t1", "<em>✓</em> Proprietari<br/>di casa", 70, 110, WHITE),
    tb("d-t2", "<em>✓</em> Entrambi<br/>i decisori", 70, 110, WHITE),
    tb("d-t3", "<em>✓</em> Almeno<br/><em>4</em> infissi", 70, 110, WHITE),
])
css = """
    .glow { position: absolute; opacity: 0; background: radial-gradient(ellipse at 50% 45%, #ffe2a8 0%, #ffb35c 55%, #e37739 100%); box-shadow: 0 0 50px 14px rgba(255,170,80,0.6); }
"""
js = """
    bg("#15110E", 0);
    tl.fromTo("#d-house", { scale: 1.35, y: 200, rotation: 6 }, { scale: 1, y: 0, rotation: 2, duration: 0.35, ease: "expo.out" }, 0);
    tl.to("#d-house", { rotation: -2, duration: 3.6, ease: "none" }, 0.35);
    [[0.04, "#d-w1", "#d-t1", 1.32], [1.32, "#d-w2", "#d-t2", 2.65], [2.65, "#d-w3", "#d-t3", 9]].forEach(([t, w, txt, end]) => {
      tl.fromTo(w, { opacity: 0 }, { opacity: 0.95, duration: 0.12, ease: "power2.out" }, t);
      punch(txt, t, 1.7); shake(t + 0.02, 10);
      if (end < 9) cut(txt, end);
    });
    tl.fromTo("#d-w4", { opacity: 0 }, { opacity: 0.95, duration: 0.12, ease: "power2.out" }, 3.04);
    shake(3.06, 14);
"""
files["s4"] = scene("s4", 3.95, body, js, css)

# ============ S5: 17.30 - 20.00 (local = abs - 17.30) =========================
body = "\n".join([
    card("e-kit", 190, 440, 700, 860, img("i5.jpg"), 51, '<i id="e-dim" class="dim"></i>'),
    tb("e-t1", "Manca un<br/><em>requisito?</em>", 120, 120, K),
    tb("e-t2", "quell'appuntamento", 170, fit("QUELL'APPUNTAMENTO", 120), WHITE),
    tb("e-t3", "Non lo<br/>paghi", 70, 190, WHITE),
])
css = """
    .dim { position: absolute; inset: 0; background: #15110E; mix-blend-mode: color; opacity: 0; }
"""
js = """
    bg("#F4F1EC", 0); bg("#15110E", 1.2); bg("#C8321E", 2.0);
    tl.fromTo("#e-kit", { x: 1000, rotation: 16 }, { x: 0, rotation: 4, duration: 0.35, ease: "expo.out" }, 0.0);
    tl.to("#e-kit", { rotation: 1, duration: 0.8, ease: "none" }, 0.35);
    punch("#e-t1", 0.07); cut("#e-t1", 1.2);
    punch("#e-t2", 1.2, 1.5); cut("#e-t2", 2.0);
    tl.to("#e-kit", { scale: 0.9, rotation: -3, y: 80, duration: 0.25, ease: "power3.out" }, 1.2);
    // NON LO PAGHI: the scene drains to black and white and shakes
    tl.fromTo("#e-dim", { opacity: 0 }, { opacity: 1, duration: 0.06, ease: "none" }, 2.0);
    punch("#e-t3", 2.0, 2.6); shake(2.02, 26);
"""
files["s5"] = scene("s5", 2.7, body, js, css)

# ============ S6 CTA: 20.00 - 24.30 (local = abs - 20.00) =====================
body = "\n".join([
    '      <div id="f-prof" class="prof"><img src="assets/img/profilo_cut.png" alt="" /></div>',
    tb("f-t1", "Clicca su", 130, fit("CLICCA SU", 150), WHITE),
    '      <div id="f-btn1" class="btn1"><span id="f-btn1in">Scopri di più</span></div>',
    tb("f-t3", "Prenota una<br/>chiamata", 90, 112, WHITE),
    '      <div id="f-btn" class="btn"><span>Chiamata conoscitiva</span><svg width="44" height="44" viewBox="0 0 46 46"><path d="M8 23 H36 M24 11 L36 23 L24 35" fill="none" stroke="#E37739" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" /></svg></div>',
    '      <div id="f-mark" class="mark">Infissi<em>Marketing</em></div>',
])
css = """
    .prof { position: absolute; left: 290px; top: 330px; width: 500px; height: 782px; transform-style: preserve-3d; filter: drop-shadow(0 40px 40px rgba(80,30,0,0.45)); }
    .prof img { width: 100%; height: 100%; object-fit: contain; }
    .btn1 { position: absolute; left: 240px; top: 330px; width: 600px; height: 130px; border-radius: 26px; background: #fff; color: #15110E; display: flex;
      align-items: center; justify-content: center; font-family: "Inter", sans-serif; font-weight: 900; font-size: 56px; z-index: 21; box-shadow: 0 20px 40px rgba(0,0,0,0.25); }
    .btn { position: absolute; left: 140px; top: 1050px; width: 800px; height: 120px; border-radius: 26px; background: #fff; color: #15110E; display: flex;
      align-items: center; justify-content: center; gap: 18px; font-family: "Inter", sans-serif; font-weight: 900; font-size: 46px; z-index: 21; box-shadow: 0 20px 40px rgba(0,0,0,0.25); }
    .mark { position: absolute; left: 0; right: 0; top: 1200px; text-align: center; font-family: "Inter", sans-serif; font-weight: 900; font-size: 60px; letter-spacing: -0.04em; color: #fff; z-index: 21; }
    .mark em { font-style: normal; color: #15110E; }
"""
js = """
    bg("#E37739", 0);
    tl.fromTo("#f-prof", { y: -1200, rotation: -25, rotationY: -40, scale: 1.2 }, { y: 0, rotation: -4, rotationY: -20, scale: 0.86, duration: 0.4, ease: "back.out(1.2)" }, 0.0);
    tl.to("#f-prof", { rotationY: 22, rotation: 2, duration: 3.8, ease: "power1.inOut" }, 0.4);
    shake(0.38, 14);
    punch("#f-t1", 0.01); cut("#f-t1", 1.44);
    // "Scopri di più" button pops and gets tapped
    tl.fromTo("#f-btn1", { opacity: 0, scale: 1.8 }, { opacity: 1, scale: 1, duration: 0.18, ease: "back.out(2)" }, 0.51);
    tl.to("#f-btn1", { scale: 0.9, duration: 0.07, ease: "power2.in" }, 0.85);
    tl.to("#f-btn1", { scale: 1.0, duration: 0.25, ease: "back.out(2.5)" }, 0.92);
    tl.to("#f-btn1", { opacity: 0, scale: 0.6, y: -60, duration: 0.18, ease: "power3.in" }, 1.36);
    punch("#f-t3", 1.44, 1.6);
    tl.fromTo("#f-btn", { opacity: 0, y: 140 }, { opacity: 1, y: 0, duration: 0.4, ease: "expo.out" }, 1.6);
    tl.fromTo("#f-mark", { opacity: 0, clipPath: "inset(0% 100% 0% 0%)" }, { opacity: 1, clipPath: "inset(0% 0% 0% 0%)", duration: 0.45, ease: "power3.out" }, 1.85);
"""
files["s6"] = scene("s6", 4.3, body, js, css)

for k, v in files.items():
    open(f"compositions/{k}.html", "w").write(v)

# ============ index ===========================================================
scenes = [("s1", 0, 4.4), ("s2", 4.28, 5.9), ("s3", 10.1, 3.25), ("s4", 13.35, 3.95), ("s5", 17.3, 2.7), ("s6", 20.0, 4.3)]
PEAK = {"pop": 0.12, "whoosh": 0.16, "whoosh-short": 0.16, "impact-bass-1": 0.07, "click": 0.07, "key-press": 0.08}
cues = [
    ("impact-bass-1", 0.86, 0.45), ("whoosh-short", 1.25, 0.3), ("pop", 2.35, 0.6), ("whoosh", 2.95, 0.35), ("impact-bass-1", 3.18, 0.5), ("pop", 3.71, 0.5),
    ("whoosh", 4.30, 0.3), ("pop", 4.89, 0.5), ("whoosh-short", 5.45, 0.3), ("pop", 6.20, 0.5), ("whoosh", 7.30, 0.3), ("pop", 7.38, 0.5),
    ("impact-bass-1", 8.12, 0.5), ("whoosh-short", 8.20, 0.25), ("pop", 9.22, 0.5),
    ("whoosh", 10.10, 0.3), ("impact-bass-1", 10.24, 0.45), ("pop", 10.51, 0.5), ("whoosh", 10.72, 0.35), ("whoosh-short", 11.62, 0.3), ("pop", 12.28, 0.5), ("pop", 12.79, 0.5),
    ("whoosh", 13.36, 0.3), ("key-press", 13.39, 2.5), ("key-press", 14.67, 2.5), ("key-press", 16.00, 2.5), ("key-press", 16.39, 2.5),
    ("whoosh-short", 17.30, 0.3), ("pop", 17.37, 0.5), ("whoosh", 18.50, 0.3), ("impact-bass-1", 19.32, 0.6),
    ("whoosh", 20.0, 0.35), ("impact-bass-1", 20.38, 0.4), ("pop", 20.51, 0.5), ("click", 20.85, 0.9), ("pop", 21.44, 0.5), ("whoosh-short", 21.6, 0.25),
]
DUR = {"pop": 0.3, "whoosh": 0.4, "whoosh-short": 0.4, "impact-bass-1": 1.4, "click": 0.3, "key-press": 0.2}
busy = {}
aud = []
for i, (f, t, v) in enumerate(cues):
    st = max(0, t - PEAK[f]); en = st + DUR[f]
    tr = 10
    while busy.get(tr, -1) > st:
        tr += 1
    busy[tr] = en
    aud.append(f'      <audio id="sfx{i:02d}" class="clip" src="assets/sfx/{f}.mp3" data-start="{st:.2f}" data-duration="{DUR[f]}" data-track-index="{tr}" data-volume="{v}"></audio>')
hosts = "\n".join(f'''      <div id="{k}" data-composition-id="{k}" data-composition-src="compositions/{k}.html"
        data-start="{st}" data-duration="{du}" data-track-index="1" data-track-kind="graphics"
        data-width="1080" data-height="1350" style="z-index:{i+1}"></div>''' for i, (k, st, du) in enumerate(scenes))
html = f"""<!doctype html>
<html lang="it">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1080, height=1350" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ margin: 0; width: 1080px; height: 1350px; overflow: hidden; background: #15110e; }}
      #root {{ position: relative; width: 100%; height: 100%; background: #15110e; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="24.3" data-width="1080" data-height="1350">
{hosts}
      <audio id="music" class="clip" src="assets/music/bed.wav" data-start="0" data-duration="24.3" data-media-start="0.5" data-track-index="4" data-volume="0.13" data-fade-out="0.8"></audio>
      <audio id="vo" class="clip" src="assets/narration.wav" data-start="0" data-duration="24.3" data-track-index="3" data-volume="1.25"></audio>
{chr(10).join(aud)}
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      window.__timelines["main"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
"""
open("index.html", "w").write(html)
print("ok", len(aud), "sfx")
