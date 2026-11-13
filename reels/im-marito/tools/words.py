import json,re,sys
# line: (offset, duration, segments[(text, seg_start, seg_end)])  times local to line
lines=[
 (0.15,3.8258,[("Il marito non era presente all'appuntamento…",0,2.091),("e quindi non ce l'hanno pagato.",2.428,3.8258)]),
 (4.10,1.7462,[("Anche se poi hanno chiuso la vendita.",0,1.7462)]),
 (6.05,4.7922,[("Da noi funziona così:",0,1.083),("zero fisso mensile.",1.393,2.578),("Paghi solo gli appuntamenti qualificati.",2.968,4.7922)]),
 (11.0,1.8663,[("Qualificati secondo i tuoi parametri.",0,1.8663)]),
 (13.05,4.6336,[("Sui nostri clienti, il marketing pesa al massimo il 3,5% del fatturato.",0,4.6336)]),
 (17.90,4.0392,[("Ci prendiamo il rischio con te.",0,1.2285),("Clicca su Scopri di più",1.6007,2.8606),("e prenota una chiamata.",3.0819,4.0392)]),
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
