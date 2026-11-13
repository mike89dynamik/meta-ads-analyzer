# SPEC — InfissiMarketing · "Il marito non c'era"

GOAL:    Il titolare di un'azienda di infissi capisce che con InfissiMarketing paga solo
         appuntamenti qualificati (zero fisso) e clicca "Scopri di più" per prenotare una chiamata.
AUDIENCE: titolari di aziende di serramenti (B2B). Angolo: rischio condiviso / pay per performance
         (da "01 Acquisition — 6. Oneshot", Agenzia_Script_Settembre_2026.docx).
OUTPUT:  HyperFrames, 1080 x 1350 (4:5 feed ads), 30 fps
TIME:    23.8 s x 30 = 714 frames. Voce = orologio (Arcads "Davide", atempo 1.07; vecchia Kokoro in assets/voice/kokoro).

LOOK:    brand: arancio #E37739, bianco #FFFFFF; inchiostro scuro #1F1B18 solo per testo su bianco
         e box sottotitoli. Fondo pieno (arancio o bianco), niente gradienti, niente glow.
         Display: Inter 900, tracking -2%, titoli 96–120 px; corpo Inter 700 40–48 px.
         Medium: interfacce reali piatte (chat WhatsApp, fattura, checklist) come carte bianche
         raggio 36 px, ombra 0 24px 60px rgba(0,0,0,.18).
OBJECTS: chat WhatsApp del cliente (header con avatar, bolle verdi/bianche, spunte blu);
         fattura agenzia con righe mensili 2.000 €; timbro "NON PAGATO €0";
         checklist appuntamento qualificato (4 parametri); barra fatturato con fetta 3,5%;
         wordmark "InfissiMarketing" + pulsante "Prenota una chiamata conoscitiva".
SAFE:    niente testo sopra y=168; tutto il testo tra 168 e 1330; sottotitoli centrati a y≈1250.

MOTION:  arrivi expo.out / power4.out 0.4–0.6 s; uscite power3.in 0.25 s; timbro: scala 1.6→1
         in 0.18 s back.out(1.6) + impatto; transizioni: colour flood 0.4 s power2.inOut,
         object-becomes (la carta resta e cambia contenuto). Titoli fermi dopo l'arrivo.

SHOTS (assoluti):
  S1 0.00–6.10  hook  fondo arancio. Frame 0: titolo "NON CE L'HANNO PAGATO" + chat già aperta.
                0.3 bolla cliente "mancava il marito…"; 2.9 timbro €0 NON PAGATO (impatto);
                4.1 seconda bolla "però abbiamo firmato" + notifica.
  S2 6.10–9.75  colour flood al bianco; la carta diventa fattura 2.000 €/mese ×3;
                7.39 "zero" → barra rossa sulle righe, 7.66 "0 € FISSO" timbro arancio;
                8.68 titolo "PAGHI SOLO GLI APPUNTAMENTI QUALIFICATI".
  S3 9.75–12.85 checklist "Appuntamento qualificato =" ; spunte a 10.75 / 11.2 / 11.65 / 12.1 (tick).
  S4 12.85–18.05 flood arancio; "≤ 3,5%" conta 0→3,5 a 15.86; barra fatturato con fetta scura 3,5%;
                nota "pubblicità + nostro compenso, sui clienti che usano il modello".
  S5 18.05–23.50 fondo bianco; "Ci prendiamo il RISCHIO con te"; 19.75 wordmark + pulsante; 20.27
                press sul pulsante (click); hold fino a 23.5.
  CAPTIONS 0.15–22.4 gruppi 2–3 parole, box scuro, parola attiva arancio.

SOUND:   voce; sfx: notification (0.3, 4.1), impact (2.9), whoosh (6.1, 12.85, 18.05),
         pop (7.66, 15.86), key-press ticks (checklist), click (20.27). Nessuna musica (nessuna
         sorgente locale): da aggiungere se disponibile.
BANNED:  fondo scuro/void, glow, gradienti viola/blu, tre card in fila, logo in angolo,
         tutto che sale in dissolvenza, titoli che si muovono dopo l'arrivo, emoji (font non garantito).
QA:      0.0 (copertina = hook), 3.2 (timbro), 5.0, 8.0, 9.2, 12.3, 16.6, 21.0, 23.3.
