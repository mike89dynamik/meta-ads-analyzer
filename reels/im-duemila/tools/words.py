import json,re,sys
lines=[
 (0.10,4.2621,[("Serramentista,",0,0.766),("paghi duemila euro al mese…",0.949,2.227),("anche quando gli appuntamenti sono da buttare?",2.472,4.2621)]),
 (4.45,4.4559,[("Solo due finestre.",0,0.966),("Casa in affitto.",1.223,1.972),("Il marito che non c'è.",2.223,3.09),("Lavori tra due anni.",3.346,4.4559)]),
 (9.00,3.9427,[("E il commerciale ha perso la giornata.",0,1.723),("Con noi,",1.977,2.393),("il fisso mensile è zero.",2.549,3.9427)]),
 (13.05,4.0107,[("Paghi solo gli appuntamenti che rispettano i tuoi parametri.",0,2.376),("Se ne manca uno,",2.602,3.293),("non lo paghi.",3.452,4.0107)]),
 (17.15,3.463,[("E il marketing",0,0.669),("pesa al massimo il 3,5% sul fatturato.",0.763,3.463)]),
 (20.70,2.3501,[("Clicca su Scopri di più,",0,1.15),("e prenota una chiamata.",1.342,2.3501)]),
]
def weight(w):
    if '3,5%' in w: return len("trevirgolacinquepercento")
    if w.startswith('duemila'): return len(w)+1.5
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
