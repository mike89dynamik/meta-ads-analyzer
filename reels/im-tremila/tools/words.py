import json,re,sys
lines=[
 (0.15,4.4992,[("Investiresti 3.500 euro…",0,2.186),("per chiudere 100.000 euro di contratti?",2.508,4.4992)]),
 (4.90,6.4502,[("È quanto pesa il marketing sui serramentisti che lavorano con noi:",0,3.122),("al massimo il 3,5% del fatturato.",3.455,6.285)]),
 (11.60,3.368,[("Zero canone mensile.",0,1.291),("Paghi solo gli appuntamenti qualificati.",1.601,3.368)]),
 (15.20,4.3007,[("Proprietari di casa.",0,1.068),("Entrambi i decisori.",1.439,2.597),("Almeno quattro infissi.",2.92,4.3007)]),
 (19.75,2.7841,[("Se manca un requisito,",0,1.098),("quell'appuntamento non lo paghi.",1.25,2.7841)]),
 (22.80,3.1566,[("Clicca su Scopri di più,",0,1.339),("e prenota una chiamata conoscitiva.",1.578,3.1566)]),
]
SPOKEN={'3.500':'tremilacinquecento','100.000':'centomila','3,5%':'trevirgolacinquepercento'}
def weight(w):
    for k,v in SPOKEN.items():
        if k in w: return len(v)
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
