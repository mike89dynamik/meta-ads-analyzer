# Generates compositions/s1..s6.html for the "3.500 euro" reel.
import json
TEAR = '''
    function tearPoly(y) {
      const n = 16; let s = "polygon(0% 100%, 100% 100%";
      for (let i = n; i >= 0; i--) {
        const x = (i / n) * 100;
        const yy = y + (i % 2 ? 2.4 : -2.0) + (((i * 37) % 5) - 2) * 0.5;
        s += `, ${x.toFixed(2)}% ${yy.toFixed(2)}%`;
      }
      return s + ")";
    }'''
BASE_CSS = '''
    #root { position: absolute; inset: 0; background: transparent; font-family: "Inter", sans-serif; overflow: hidden; }
    .plate { position: absolute; inset: 0; overflow: hidden; }
    .plate img, .plate video { position: absolute; left: 0; top: 0; width: 1080px; height: 1350px; object-fit: cover; }
    .scrim { position: absolute; inset: 0; background: linear-gradient(180deg, rgba(10,8,6,0.62) 0%, rgba(10,8,6,0.0) 34%, rgba(10,8,6,0.0) 66%, rgba(10,8,6,0.55) 100%); }
    .t { position: absolute; left: 40px; right: 40px; text-align: center; color: #fff; font-weight: 900; letter-spacing: -0.035em; line-height: 0.95;
         text-transform: uppercase; text-shadow: 0 6px 30px rgba(0,0,0,0.45); }
    .t em { font-style: normal; color: #e37739; }
'''
def scene(sid, dur, css, body, js):
    return f'''<template>
  <style>{BASE_CSS}{css}
  </style>
  <div id="root" data-composition-id="{sid}" data-width="1080" data-height="1350">
    <div id="{sid}-wrap" style="position:absolute;inset:0" data-layout-allow-overflow data-layout-allow-overlap>
{body}
    </div>
  </div>
  <script>{TEAR}
    const tl = gsap.timeline({{ paused: true }});
    function tearTo(sel, at, dur) {{
      const el = document.querySelector(sel); const p = {{ y: 112 }};
      el.style.clipPath = tearPoly(112);
      tl.fromTo(p, {{ y: 112 }}, {{ y: -8, duration: dur, ease: "power3.inOut", onUpdate: () => {{ el.style.clipPath = tearPoly(p.y); }} }}, at);
    }}
{js}
    window.__timelines["{sid}"] = tl;
  </script>
</template>
'''
files = {}
# ---------------- S1: hook (0 - 4.8)
files['s1'] = scene('s1', 5.3, '''
    #s1-a { z-index: 1; } #s1-b { z-index: 2; }
    #s1-flash { position: absolute; inset: 0; background: #fff; z-index: 3; opacity: 0; }
    #s1-q { top: 90px; font-size: 76px; z-index: 5; }
    #s1-n1 { top: 190px; font-size: 230px; z-index: 5; }
    #s1-n2 { top: 150px; font-size: 200px; z-index: 5; color: #e37739; }
    #s1-n2s { top: 360px; font-size: 76px; z-index: 5; }
''', '''      <div id="s1-a" class="plate"><div id="s1-av"><video id="s1-vid" class="clip" src="assets/video/v1.mp4" data-start="0" data-duration="2.75" muted playsinline></video></div><div class="scrim"></div></div>
      <div id="s1-b" class="plate"><div id="s1-bk"><img src="assets/img/i7.jpg" alt="" /></div><div class="scrim"></div></div>
      <div id="s1-flash"></div>
      <div id="s1-q" class="t">Investiresti</div>
      <div id="s1-n1" class="t">3.500 €</div>
      <div id="s1-n2" class="t">100.000 €</div>
      <div id="s1-n2s" class="t">di contratti?</div>''', '''
    tl.fromTo("#s1-av", { scale: 1.0 }, { scale: 1.08, duration: 2.8, ease: "none" }, 0);
    tl.fromTo("#s1-q", { opacity: 1, scale: 1.12 }, { opacity: 1, scale: 1, duration: 0.4, ease: "back.out(1.6)" }, 0);
    tl.fromTo("#s1-n1", { opacity: 0, scale: 2.6, filter: "blur(14px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.3, ease: "back.out(1.5)" }, 0.92);
    // flash cut to the signed contracts
    tl.fromTo("#s1-b", { opacity: 0 }, { opacity: 1, duration: 0.01 }, 2.62);
    tl.fromTo("#s1-flash", { opacity: 0 }, { opacity: 0.85, duration: 0.06, ease: "none" }, 2.56);
    tl.to("#s1-flash", { opacity: 0, duration: 0.25, ease: "power2.out" }, 2.63);
    tl.fromTo("#s1-bk", { scale: 1.18, rotation: -2 }, { scale: 1.0, rotation: 0, duration: 2.2, ease: "power2.out" }, 2.62);
    tl.to("#s1-q, #s1-n1", { opacity: 0, filter: "blur(10px)", duration: 0.15, ease: "power3.in" }, 2.5);
    tl.fromTo("#s1-n2", { opacity: 0, scale: 2.4, filter: "blur(14px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.3, ease: "back.out(1.5)" }, 3.28);
    tl.fromTo("#s1-n2s", { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.3, ease: "expo.out" }, 3.95);''')

# ---------------- S2: showroom, 3.5% (4.8 - 11.4)
files['s2'] = scene('s2', 7.0, '''
    #s2-tear { position: absolute; inset: 0; }
    #s2-h { top: 90px; font-size: 70px; }
    #s2-n { top: 330px; font-size: 300px; letter-spacing: -0.06em; }
    #s2-n .le { font-size: 190px; vertical-align: 0.18em; margin-right: 14px; }
    #s2-s { top: 640px; font-size: 64px; }
''', '''      <div id="s2-tear">
        <div class="plate"><div id="s2-av"><video id="s2-vid" class="clip" src="assets/video/v2.mp4" data-start="0" data-duration="6.6" data-playback-rate="0.75" muted playsinline></video></div><div class="scrim"></div></div>
        <div id="s2-h" class="t">Il marketing pesa</div>
        <div id="s2-n" class="t"><span id="s2-nin"><span class="le">≤</span><em>3,5%</em></span></div>
        <div id="s2-s" class="t">del fatturato</div>
      </div>''', '''
    tearTo("#s2-tear", 0, 0.45);
    tl.fromTo("#s2-av", { scale: 1.12 }, { scale: 1.0, duration: 6.6, ease: "none" }, 0);
    // "marketing" abs 5.73 -> 0.93
    tl.fromTo("#s2-h", { opacity: 0, filter: "blur(10px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.2, ease: "power2.out" }, 0.9);
    // "3,5%" abs 9.16 -> 4.36
    tl.fromTo("#s2-nin", { opacity: 0, scale: 2.6, filter: "blur(16px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.32, ease: "back.out(1.5)" }, 4.33);
    tl.fromTo("#s2-s", { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.3, ease: "expo.out" }, 5.8);''')

# ---------------- S3: installer, zero canone (11.4 - 15.0)
files['s3'] = scene('s3', 4.0, '''
    #s3-in { position: absolute; inset: 0; }
    #s3-z { top: 100px; font-size: 240px; letter-spacing: -0.06em; }
    #s3-zs { top: 330px; font-size: 64px; }
    #s3-q { top: 100px; font-size: 84px; }
''', '''      <div id="s3-in">
        <div class="plate"><div id="s3-av"><video id="s3-vid" class="clip" src="assets/video/v3.mp4" data-start="0" data-duration="3.6" muted playsinline></video></div><div class="scrim"></div></div>
        <div id="s3-z" class="t"><em>0 €</em></div>
        <div id="s3-zs" class="t">di canone mensile</div>
        <div id="s3-q" class="t">Paghi solo<br /><em>appuntamenti</em><br />qualificati</div>
      </div>''', '''
    // whip in from the right
    tl.fromTo("#s3-in", { x: 1080, filter: "blur(18px)" }, { x: 0, filter: "blur(0px)", duration: 0.35, ease: "expo.out" }, 0);
    tl.fromTo("#s3-av", { scale: 1.0 }, { scale: 1.06, duration: 3.6, ease: "none" }, 0);
    tl.fromTo("#s3-z", { opacity: 0, scale: 2.4, filter: "blur(14px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.3, ease: "back.out(1.5)" }, 0.2);
    tl.fromTo("#s3-zs", { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.3, ease: "expo.out" }, 0.55);
    tl.to("#s3-z, #s3-zs", { opacity: 0, filter: "blur(10px)", duration: 0.15, ease: "power3.in" }, 1.7);
    // "Paghi" abs 13.20 -> 1.80
    tl.fromTo("#s3-q", { opacity: 0, filter: "blur(10px)", scale: 1.15 }, { opacity: 1, filter: "blur(0px)", scale: 1, duration: 0.25, ease: "expo.out" }, 1.8);''')

# ---------------- S4: facade, windows light up (15.0 - 19.6)
WIN = [(240,316,152,226),(688,316,152,226),(240,812,152,221),(688,812,152,221)]
wins = '\n'.join(f'        <i id="s4-w{i+1}" class="s4-w" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px"></i>' for i,(x,y,w,h) in enumerate(WIN))
files['s4'] = scene('s4', 5.1, '''
    #s4-in { position: absolute; inset: 0; }
    .s4-w { position: absolute; background: radial-gradient(ellipse at 50% 45%, #ffe2a8 0%, #ffb35c 55%, #e37739 100%); box-shadow: 0 0 60px 18px rgba(255, 170, 80, 0.55); }
    #s4-list { position: absolute; left: 0; right: 0; top: 40px; text-align: center; }
    .s4-row { display: block; margin: 0 auto 12px; color: #fff; font-weight: 900; font-size: 56px; letter-spacing: -0.03em; text-transform: uppercase; text-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    .s4-row b { display: inline-block; width: 56px; height: 56px; margin-right: 18px; border-radius: 14px; background: #e37739; vertical-align: -6px; position: relative; }
    .s4-row b:after { content: ""; position: absolute; left: 18px; top: 8px; width: 14px; height: 28px; border: solid #fff; border-width: 0 7px 7px 0; transform: rotate(45deg); }
''', f'''      <div id="s4-in">
        <div class="plate"><div id="s4-av"><img src="assets/img/i4.jpg" alt="" />
{wins}
        </div></div>
        <div id="s4-list">
          <span id="s4-r1" class="s4-row"><b></b>Proprietari di casa</span>
          <span id="s4-r2" class="s4-row"><b></b>Entrambi i decisori</span>
          <span id="s4-r3" class="s4-row"><b></b>Almeno 4 infissi</span>
        </div>
      </div>''', '''
    tl.fromTo("#s4-in", { opacity: 0, scale: 1.15, filter: "blur(16px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.4, ease: "expo.out" }, 0);
    tl.fromTo("#s4-av", { scale: 1.0 }, { scale: 1.07, duration: 4.6, ease: "none", transformOrigin: "50% 60%" }, 0);
    // lights on the spoken params: abs 15.20 16.64 18.12, all four on "quattro" 18.54
    [[0.2, "#s4-w1", "#s4-r1"], [1.64, "#s4-w2", "#s4-r2"], [3.12, "#s4-w3", "#s4-r3"]].forEach(([t, w, r]) => {
      tl.fromTo(w, { opacity: 0 }, { opacity: 0.92, duration: 0.18, ease: "power2.out" }, t);
      tl.fromTo(r, { opacity: 0, scale: 1.4, filter: "blur(8px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.25, ease: "back.out(1.6)" }, t);
    });
    tl.fromTo("#s4-w4", { opacity: 0 }, { opacity: 0.92, duration: 0.18, ease: "power2.out" }, 3.54);''')

# ---------------- S5: kitchen, non lo paghi (19.6 - 22.7)
files['s5'] = scene('s5', 3.5, '''
    #s5-in { position: absolute; inset: 0; }
    #s5-q { top: 90px; font-size: 84px; }
    #s5-stamp { position: absolute; left: 70px; top: 800px; padding: 22px 40px; border: 14px solid #c8321e; border-radius: 26px; background: rgba(255,255,255,0.92);
      color: #c8321e; font-weight: 900; font-size: 104px; letter-spacing: -0.03em; line-height: 0.95; text-align: center; }
''', '''      <div id="s5-in">
        <div class="plate"><div id="s5-av"><img src="assets/img/i5.jpg" alt="" /></div><div class="scrim"></div></div>
        <div id="s5-q" class="t">Manca un<br /><em>requisito?</em></div>
        <div id="s5-stamp" data-layout-allow-occlusion>NON LO<br />PAGHI</div>
      </div>''', '''
    tearTo("#s5-in", 0, 0.42);
    tl.fromTo("#s5-av", { scale: 1.12, x: 30 }, { scale: 1.0, x: 0, duration: 3.1, ease: "none" }, 0);
    tl.fromTo("#s5-q", { opacity: 0, filter: "blur(10px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.2, ease: "power2.out" }, 0.18);
    // "paghi" abs 22.24 -> 2.64
    tl.fromTo("#s5-stamp", { opacity: 0, scale: 2.4, rotation: -14 }, { opacity: 1, scale: 1, rotation: -7, duration: 0.2, ease: "back.out(1.6)" }, 2.3);''')

# ---------------- S6: CTA (22.7 - 27.6)
files['s6'] = scene('s6', 4.9, '''
    #s6-in { position: absolute; inset: 0; background: #db6e33; }
    #s6-mark { position: absolute; left: 0; right: 0; top: 70px; text-align: center; font-weight: 900; font-size: 76px; letter-spacing: -0.04em; color: #fff; }
    #s6-mark em { font-style: normal; color: #1f1b18; }
    #s6-btn { position: absolute; left: 110px; top: 1030px; width: 860px; height: 140px; border-radius: 28px; background: #fff; color: #1f1b18;
      display: flex; align-items: center; justify-content: center; gap: 20px; font-size: 48px; font-weight: 900; box-shadow: 0 20px 50px rgba(0,0,0,0.25); }
''', '''      <div id="s6-in">
        <div class="plate"><div id="s6-av"><img src="assets/img/i6.jpg" alt="" /></div></div>
        <div id="s6-mark"><span id="s6-markin">Infissi<em>Marketing</em></span></div>
        <div id="s6-btn"><span>Prenota una chiamata</span><svg width="48" height="48" viewBox="0 0 46 46"><path d="M8 23 H36 M24 11 L36 23 L24 35" fill="none" stroke="#e37739" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" /></svg></div>
      </div>''', '''
    tl.fromTo("#s6-in", { opacity: 0, scale: 1.2, filter: "blur(18px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.4, ease: "expo.out" }, 0);
    tl.fromTo("#s6-av", { scale: 1.12, rotation: -3 }, { scale: 1.0, rotation: 0, duration: 4.9, ease: "power1.out" }, 0);
    tl.fromTo("#s6-markin", { opacity: 0, clipPath: "inset(0% 100% 0% 0%)" }, { opacity: 1, clipPath: "inset(0% 0% 0% 0%)", duration: 0.45, ease: "power3.out" }, 0.1);
    tl.fromTo("#s6-btn", { opacity: 0, y: 80, scale: 0.94 }, { opacity: 1, y: 0, scale: 1, duration: 0.45, ease: "expo.out" }, 0.4);
    // tap on "Scopri" abs 23.36 -> 0.66
    tl.to("#s6-btn", { scale: 0.94, duration: 0.08, ease: "power2.in" }, 0.9);
    tl.to("#s6-btn", { scale: 1, duration: 0.3, ease: "back.out(2)" }, 0.98);''')
for k, v in files.items():
    open(f'compositions/{k}.html', 'w').write(v)
print('ok')
