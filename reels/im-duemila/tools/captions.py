import json,sys,re,html
proj=sys.argv[1]
W=json.load(open(f"{proj}/assets/words.json"))
SIZES=[1,2,3,2,2,3,3,3,2,3,4,3,2,2,2,3,2,2,2,2,3,4,3,3,3,2,2,2,3,2,2]
assert sum(SIZES)==len(W),(sum(SIZES),len(W))
groups=[];k=0
for n in SIZES:
    groups.append(W[k:k+n]);k+=n
def disp(t):
    t=re.sub(r"[.:]$","",t)
    return t
out=[];js=[]
for gi,g in enumerate(groups):
    s=max(0.1,g[0]["start"]-0.05)
    nxt=groups[gi+1][0]["start"]-0.05 if gi+1<len(groups) else 24.7
    e=min(g[-1]["end"]+0.25,nxt)
    e=round(e,3);s=round(s,3)
    spans=" ".join(f'<span class="cw" id="cw-{w["id"]}">{html.escape(disp(w["text"]))}</span>' for w in g)
    out.append(f'    <div id="cg-{gi}" class="cg clip" data-start="{s}" data-duration="{round(e-s,3)}"><span class="cbox">{spans}</span></div>')
    js.append(f'    tl.fromTo("#cg-{gi} .cbox", {{ opacity: 0, scale: 0.92 }}, {{ opacity: 1, scale: 1, duration: 0.12, ease: "back.out(2)" }}, {s});')
    for w in g:
        js.append(f'    tl.set("#cw-{w["id"]}", {{ color: "#E37739" }}, {round(w["start"],3)}); tl.set("#cw-{w["id"]}", {{ color: "#FFFFFF" }}, {round(w["end"],3)});')
    js.append(f'    tl.to("#cg-{gi} .cbox", {{ opacity: 0, duration: 0.1, ease: "power2.in" }}, {round(e-0.1,3)});')
doc=f'''<template>
  <style>
    #root {{ position: absolute; inset: 0; pointer-events: none; }}
    .cg {{ position: absolute; left: 0; right: 0; top: 1125px; text-align: center; overflow: visible; }}
    .cbox {{ display: inline-block; padding: 14px 30px 18px; border-radius: 20px; background: #1F1B18;
      font-family: "Inter", sans-serif; font-weight: 900; font-size: 56px; line-height: 1.1; letter-spacing: -0.01em; color: #FFFFFF; }}
    .cw {{ color: #FFFFFF; }}
  </style>
  <div id="root" data-composition-id="captions" data-width="1080" data-height="1350">
{chr(10).join(out)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
{chr(10).join(js)}
    window.__timelines["captions"] = tl;
  </script>
</template>
'''
open(f"{proj}/compositions/captions.html","w").write(doc)
print(len(groups),"groups");[print(round(g[0]['start'],2)," ".join(w['text'] for w in g)) for g in groups]
