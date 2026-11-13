import json,re,sys
# line: (offset, duration, segments[(text, seg_start, seg_end)])  times local to line
lines=[
 (0.15,3.8165,[("Il marito non era presente all'appuntamento…",0,2.04),("e quindi non ce l'hanno pagato.",2.30,3.8165)]),
 (4.10,1.8787,[("Anche se poi hanno chiuso la vendita.",0,1.8787)]),
 (6.25,4.3482,[("Da noi funziona così: zero fisso mensile.",0,2.148),("Paghi solo gli appuntamenti qualificati.",2.425,4.3482)]),
 (10.75,1.9205,[("Qualificati secondo i tuoi parametri.",0,1.9205)]),
 (12.90,5.0375,[("Sui nostri clienti, il marketing pesa al massimo il 3,5% del fatturato.",0,5.0375)]),
 (18.15,4.0073,[("Ci prendiamo il rischio con te.",0,1.414),("Clicca su Scopri di più e prenota una chiamata.",1.604,4.0073)]),
]
# spoken weight: 3,5% is spoken as "tre virgola cinque per cento"
def weight(w):
    if '3,5%' in w: return len("trevirgolacinquepercento")
    return len(re.sub(r"[^\wàèéìòù']","",w))+1.5
out=[];i=0
for li,(off,dur,segs) in enumerate(lines):
    for text,s,e in segs:
        ws=text.split(); tot=sum(weight(w) for w in ws); t=s
        for w in ws:
            d=(e-s)*weight(w)/tot
            out.append({"id":f"w{i}","line":li,"text":w,"start":round(off+t,3),"end":round(off+t+d,3)}); t+=d; i+=1
json.dump(out,open(sys.argv[1],"w"),ensure_ascii=False,indent=0)
print(len(out))
