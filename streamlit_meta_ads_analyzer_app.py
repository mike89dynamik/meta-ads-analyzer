# -*- coding: utf-8 -*-
"""
Streamlit Meta Ads Analyzer (IT)
--------------------------------
Mini web-app per analisi giornaliera delle metriche Meta Ads.

File: app.py
Requisiti: streamlit, pandas, numpy, plotly, scikit-learn, python-dateutil

Note principali:
- UX in italiano, tema chiaro
- Sidebar: filtri (azienda, livello, entità), date picker (giorno o intervallo), valori globali opzionali,
  import/export CSV
- Header: breadcrumb + KPI snapshot
- Tabella editing dati (solo righe per entità selezionata) con Aggiungi/Elimina riga
- Tabs per metrica (CPL, CPC, CPC link, CTR all, CTR link, Frequenza, Impression, Copertura, CPM)
- Tab "Degradamento": evidenzia metriche oltre soglia con var% e giorni consecutivi di peggioramento
- Tab "Salute & Azioni": stato salute, raccomandazioni da motore regole, proiezioni (regressione lineare)
- Tab "Budget & Obiettivi": calcoli su lead/mese attesi e budget consigliato

CSV: i CTR sono espressi in percentuale (es. 1.25 = 1.25%)
Valuta: Euro
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

# =============================
# CONFIG
# =============================
CONFIG = {
    "APP_TITLE": "Analisi Meta Ads (IT)",
    "THEME": "light",  # Streamlit imposta il tema da settings; lasciamo etichette chiare
    "DATE_FMT": "%Y-%m-%d",
    "SPARKLINE_HEIGHT": 40,
    "THRESHOLDS": {
        "ctr_link_var_pct_neg": -15.0,  # calo < -15%
        "cpm_var_pct_pos": 15.0,        # aumento > +15%
        "freq_increase_abs": 0.5,       # aumento assoluto > 0.5
        "cpl_var_pct_pos": 15.0,        # aumento > +15%
    },
    "HEALTH_COLORS": {
        "ok": "#22c55e",       # verde
        "warn": "#f59e0b",     # giallo
        "crit": "#ef4444",     # rosso
        "muted": "#6b7280",    # grigio
    },
}

# =============================
# COLONNE & TIPI
# =============================
COLUMNS = [
    "azienda", "livello", "nome_entita", "data",
    "numero_lead_desiderati_mese", "budget_adv_disponibile_mese",
    "budget_adv_giornaliero", "cpl", "cpc", "cpc_link", "ctr_all", "ctr_link",
    "frequenza", "impression", "copertura", "cpm",
]

DTYPES = {
    "azienda": "string",
    "livello": "string",
    "nome_entita": "string",
    "data": "string",  # verrà convertita a datetime
    "numero_lead_desiderati_mese": "float64",  # opzionale
    "budget_adv_disponibile_mese": "float64",  # opzionale
    "budget_adv_giornaliero": "float64",
    "cpl": "float64",
    "cpc": "float64",
    "cpc_link": "float64",
    "ctr_all": "float64",   # percentuale (1.25 = 1.25%)
    "ctr_link": "float64",  # percentuale
    "frequenza": "float64",
    "impression": "Int64",
    "copertura": "Int64",
    "cpm": "float64",
}

LIVELLI_VALIDI = ["campagna", "adset", "inserzione"]

# =============================
# UTILS
# =============================

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Allinea colonne, tipi e ordina le colonne secondo lo schema richiesto."""
    # Aggiungi eventuali colonne mancanti
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype=DTYPES.get(col, "object"))
    # Mantieni solo le colonne previste e cast tipi
    df = df[COLUMNS].copy()
    for col, dt in DTYPES.items():
        try:
            if dt == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dt == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif col == "data":
                # manterremo come string in archivio, ma creeremo colonna datetime on the fly
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype("string")
        except Exception:
            pass
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
    return df


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.2f}%"


def fmt_eur(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"€{x:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_int(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{int(x):,}".replace(",", ".")


def variation_pct(first: Optional[float], last: Optional[float]) -> Optional[float]:
    if first is None or last is None or pd.isna(first) or pd.isna(last):
        return None
    if first == 0:
        return None
    return (last - first) / abs(first) * 100.0


def consecutive_moves(series: pd.Series, direction: str = "down") -> int:
    """Conta i giorni consecutivi di movimento (down=calo, up=aumento) alla fine della serie.
    Ritorna la lunghezza della run terminale (es. ultimi 4 giorni in calo consecutivo).
    """
    s = series.dropna()
    if len(s) < 2:
        return 0
    diffs = np.diff(s.values)
    count = 0
    if direction == "down":
        for d in diffs[::-1]:
            if d < 0:
                count += 1
            else:
                break
    else:
        for d in diffs[::-1]:
            if d > 0:
                count += 1
            else:
                break
    return int(count)


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# =============================
# DATA INIT (Session State)
# =============================
@st.cache_data(show_spinner=False)
def load_example_csv() -> pd.DataFrame:
    """Carica dati_esempio.csv se presente su disco, altrimenti DataFrame vuoto.
    L'utente può importare manualmente.
    """
    try:
        df = pd.read_csv("dati_esempio.csv", dtype=DTYPES)
        return ensure_schema(df)
    except Exception:
        return ensure_schema(pd.DataFrame(columns=COLUMNS))


def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = load_example_csv()
    if "azienda_list" not in st.session_state:
        az_list = sorted(list(st.session_state.df["azienda"].dropna().unique()))
        st.session_state.azienda_list = az_list or ["Azienda A"]
    if "global_lead_mese" not in st.session_state:
        st.session_state.global_lead_mese = None
    if "global_budget_mese" not in st.session_state:
        st.session_state.global_budget_mese = None
    if "cpl_target" not in st.session_state:
        st.session_state.cpl_target = None


# =============================
# SIDEBAR UI
# =============================

def sidebar_filters(df: pd.DataFrame) -> Tuple[str, str, str, Tuple[date, date]]:
    st.sidebar.header("Filtri")

    # Azienda select + aggiungi nuova
    azienda_sel = st.sidebar.selectbox("Azienda", options=sorted(list(set(st.session_state.azienda_list + ["➕ Aggiungi nuova…"]))))
    if azienda_sel == "➕ Aggiungi nuova…":
        nuova = st.sidebar.text_input("Nome nuova azienda", key="new_azienda")
        if nuova:
            if nuova not in st.session_state.azienda_list:
                st.session_state.azienda_list.append(nuova)
            azienda_sel = nuova
    # Filtro livelli
    livello_sel = st.sidebar.selectbox("Livello", options=LIVELLI_VALIDI, index=0)

    # Entità dinamiche
    df_az = df[df["azienda"] == azienda_sel]
    entita_opts = sorted(df_az[df_az["livello"] == livello_sel]["nome_entita"].dropna().unique().tolist())
    entita_opts = entita_opts or ["Tutte"]
    entita_sel = st.sidebar.selectbox("Nome entità", options=entita_opts)

    # Date picker: giorno singolo o intervallo
    st.sidebar.markdown("### Intervallo date")
    df_dates = parse_dates(df_az)
    min_d = pd.Timestamp.today() - pd.Timedelta(days=29)
    max_d = pd.Timestamp.today()
    if not df_dates["data_dt"].dropna().empty:
        min_d = df_dates["data_dt"].min()
        max_d = df_dates["data_dt"].max()
    range_default = (min_d.date(), max_d.date())
    date_range = st.sidebar.date_input("Seleziona intervallo", value=range_default)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    st.sidebar.markdown("---")
    st.sidebar.subheader("Valori globali opzionali")
    gl_lead = st.sidebar.number_input("Lead desiderati al mese (globale)", min_value=0, value=st.session_state.global_lead_mese or 0, step=1)
    st.session_state.global_lead_mese = gl_lead if gl_lead > 0 else None
    gl_budget = st.sidebar.number_input("Budget adv disponibile al mese (globale) €", min_value=0.0, value=st.session_state.global_budget_mese or 0.0, step=100.0, format="%0.2f")
    st.session_state.global_budget_mese = gl_budget if gl_budget > 0 else None
    cpl_target = st.sidebar.number_input("Target CPL opzionale (€)", min_value=0.0, value=st.session_state.cpl_target or 0.0, step=0.5, format="%0.2f")
    st.session_state.cpl_target = cpl_target if cpl_target > 0 else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Import/Export")
    uploaded = st.sidebar.file_uploader("Importa CSV (UTF-8, ,)", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, dtype=DTYPES)
            st.session_state.df = ensure_schema(df_up)
            st.sidebar.success("CSV importato correttamente.")
        except Exception as e:
            st.sidebar.error(f"Errore import: {e}")

    csv_buf = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Esporta CSV", data=csv_buf, file_name="export_meta_metrics.csv", mime="text/csv")

    return azienda_sel, livello_sel, entita_sel, (start_date, end_date)


# =============================
# FILTER & EDITING
# =============================

def filter_df(df: pd.DataFrame, azienda: str, livello: str, entita: str, drange: Tuple[date, date]) -> pd.DataFrame:
    d0, d1 = drange
    dmask = (pd.to_datetime(df["data"], errors="coerce").dt.date >= d0) & (pd.to_datetime(df["data"], errors="coerce").dt.date <= d1)
    mask = (df["azienda"] == azienda) & (df["livello"] == livello)
    if entita != "Tutte":
        mask &= (df["nome_entita"] == entita)
    return df[mask & dmask].copy()


def editing_block(df_entita: pd.DataFrame, azienda: str, livello: str, entita: str):
    st.markdown("## Inserimento / Editing dati")
    st.caption("Modifica solo le righe dell'entità selezionata. Usa i pulsanti per aggiungere o eliminare righe.")

    editable_cols = [c for c in COLUMNS if c not in ("azienda", "livello", "nome_entita")]

    # Aggiungi riga
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Aggiungi riga"):
            new_row = {c: None for c in COLUMNS}
            new_row["azienda"] = azienda
            new_row["livello"] = livello
            new_row["nome_entita"] = entita if entita != "Tutte" else ""
            # default data = oggi
            new_row["data"] = datetime.today().strftime(CONFIG["DATE_FMT"])
            st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
            st.experimental_rerun()
    with col_b:
        st.write("")

    # Crea una colonna di selezione per eliminare
    df_entita = df_entita.copy()
    df_entita.insert(0, "_seleziona", False)

    edited = st.data_editor(
        df_entita,
        num_rows="dynamic",
        disabled=["azienda", "livello"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "_seleziona": st.column_config.CheckboxColumn("Seleziona", help="Spunta per eliminare riga"),
            "data": st.column_config.TextColumn("Data (YYYY-MM-DD)"),
            "ctr_all": st.column_config.NumberColumn("CTR all (%)"),
            "ctr_link": st.column_config.NumberColumn("CTR link (%)"),
            "cpm": st.column_config.NumberColumn("CPM (€)"),
            "cpl": st.column_config.NumberColumn("CPL (€)"),
            "cpc": st.column_config.NumberColumn("CPC (€)"),
            "cpc_link": st.column_config.NumberColumn("CPC link (€)"),
            "budget_adv_giornaliero": st.column_config.NumberColumn("Budget giornaliero (€)"),
            "numero_lead_desiderati_mese": st.column_config.NumberColumn("Lead desiderati/mese (riga)"),
            "budget_adv_disponibile_mese": st.column_config.NumberColumn("Budget mensile (riga) €"),
            "frequenza": st.column_config.NumberColumn("Frequenza"),
            "impression": st.column_config.NumberColumn("Impression"),
            "copertura": st.column_config.NumberColumn("Copertura"),
        },
        key="editor_entita",
    )

    # Salva modifiche su session_state.df
    if st.button("Salva modifiche"):
        # Rimuovi righe selezionate
        to_delete_keys = edited[edited["_seleziona"]].index
        if len(to_delete_keys) > 0:
            # individua righe nel df base usando valori chiave
            base = st.session_state.df
            # creiamo una chiave per match (azienda, livello, nome_entita, data, impression)
            edited["_key"] = (
                edited["azienda"].astype(str)
                + "|" + edited["livello"].astype(str)
                + "|" + edited["nome_entita"].astype(str)
                + "|" + edited["data"].astype(str)
                + "|" + edited["impression"].astype(str)
            )
            base["_key"] = (
                base["azienda"].astype(str)
                + "|" + base["livello"].astype(str)
                + "|" + base["nome_entita"].astype(str)
                + "|" + base["data"].astype(str)
                + "|" + base["impression"].astype(str)
            )
            keys_to_delete = set(edited.loc[to_delete_keys, "_key"].tolist())
            st.session_state.df = base[~base["_key"].isin(keys_to_delete)].drop(columns=["_key"])  # type: ignore
        else:
            # aggiorna righe (senza colonna _seleziona)
            edited2 = edited.drop(columns=["_seleziona"]).copy()
            # merge su chiave multipla
            base = st.session_state.df
            idx = (base["azienda"].isin(edited2["azienda"])) & (base["livello"].isin(edited2["livello"]))
            if entita != "Tutte":
                idx &= base["nome_entita"].isin(edited2["nome_entita"])  # type: ignore
            # rimpiazza quelle righe con edited2 (per semplicità)
            st.session_state.df = pd.concat([base[~idx], edited2], ignore_index=True)
        st.success("Modifiche salvate.")
        st.experimental_rerun()


# =============================
# KPI SNAPSHOT
# =============================

def kpi_snapshot(df_range: pd.DataFrame):
    st.markdown("### KPI nel periodo selezionato")
    if df_range.empty:
        st.info("Nessun dato nel range selezionato.")
        return

    df_sorted = df_range.sort_values("data")
    def _agg(metric):
        s = pd.to_numeric(df_sorted[metric], errors="coerce")
        return s.mean(), s.min(), s.max(), variation_pct(s.iloc[0] if len(s)>0 else None, s.iloc[-1] if len(s)>0 else None)

    # Spesa totale = somma budget_adv_giornaliero (fallback: cpm*impr/1000 se budget mancante?)
    # Qui usiamo il budget giornaliero se presente, altrimenti stima da CPM*impr/1000
    spese = pd.to_numeric(df_sorted["budget_adv_giornaliero"], errors="coerce")
    fallback_spese = (pd.to_numeric(df_sorted["cpm"], errors="coerce") * pd.to_numeric(df_sorted["impression"], errors="coerce") / 1000.0)
    spese = spese.fillna(fallback_spese)
    spesa_tot = spese.sum()

    cpl_m, cpl_min, cpl_max, cpl_var = _agg("cpl")
    ctr_l_m, ctr_l_min, ctr_l_max, ctr_l_var = _agg("ctr_link")
    cpm_m, _, _, cpm_var = _agg("cpm")
    freq_m, _, _, freq_var = _agg("frequenza")

    # Lead totali stimati = spesa_tot / CPL medio (se disponibile)
    lead_stimati = None
    warning = None
    if cpl_m and not pd.isna(cpl_m) and cpl_m > 0:
        lead_stimati = spesa_tot / cpl_m
    else:
        warning = "Dati CPL incompleti: impossibile stimare i lead totali."

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CPL medio", fmt_eur(cpl_m), (f"var {fmt_pct(cpl_var)}" if cpl_var is not None else "N/A"))
        st.caption(f"min {fmt_eur(cpl_min)} · max {fmt_eur(cpl_max)}")
    with col2:
        st.metric("CTR link medio", fmt_pct(ctr_l_m), (f"var {fmt_pct(ctr_l_var)}" if ctr_l_var is not None else "N/A"))
        st.caption(f"min {fmt_pct(ctr_l_min)} · max {fmt_pct(ctr_l_max)}")
    with col3:
        st.metric("CPM medio", fmt_eur(cpm_m), (f"var {fmt_pct(cpm_var)}" if cpm_var is not None else "N/A"))
    with col4:
        st.metric("Frequenza media", f"{freq_m:.2f}" if not pd.isna(freq_m) else "N/A", (f"var {fmt_pct(freq_var)}" if freq_var is not None else "N/A"))
    with col5:
        st.metric("Lead totali stimati", fmt_int(lead_stimati) if lead_stimati is not None else "N/A")
        if warning:
            st.warning(warning)


# =============================
# CHARTS & STATS BY METRIC
# =============================

def metric_tab(df_range: pd.DataFrame, metric: str, title: str, is_pct: bool = False, is_currency: bool = False):
    st.subheader(title)
    if df_range.empty:
        st.info("Nessun dato nel periodo.")
        return

    dfm = df_range.sort_values("data_dt")
    y = pd.to_numeric(dfm[metric], errors="coerce")

    # Grafico linea
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfm["data_dt"], y=y, mode="lines+markers", name=title))

    # Toggle media mobile 7g
    show_ma = st.checkbox("Mostra media mobile 7g", value=True, key=f"ma_{metric}")
    if show_ma:
        fig.add_trace(go.Scatter(x=dfm["data_dt"], y=moving_average(y), mode="lines", name="Media mobile 7g", line=dict(dash="dash")))

    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Data", yaxis_title=("%" if is_pct else ("€" if is_currency else "Valore")))
    st.plotly_chart(fig, use_container_width=True)

    # Statistiche
    first = y.iloc[0] if len(y) > 0 else None
    last = y.iloc[-1] if len(y) > 0 else None
    varp = variation_pct(first, last)
    stat = {
        "media": y.mean(),
        "mediana": y.median(),
        "min": y.min(),
        "max": y.max(),
        "variazione_%": varp,
        "count_giorni": int(y.count()),
    }

    # Sparkline semplice (riusa plotly mini)
    spark = go.Figure(go.Scatter(x=list(range(len(y))), y=y, mode="lines"))
    spark.update_layout(height=CONFIG["SPARKLINE_HEIGHT"], margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))

    col_a, col_b = st.columns([3, 1])
    with col_a:
        df_stat = pd.DataFrame([stat])
        # Formattazioni user-facing
        if is_pct:
            df_stat_display = df_stat.copy()
            for c in ["media", "mediana", "min", "max", "variazione_%"]:
                df_stat_display[c] = df_stat_display[c].apply(lambda v: fmt_pct(v) if not pd.isna(v) else "N/A")
            st.dataframe(df_stat_display, hide_index=True, use_container_width=True)
        elif is_currency:
            df_stat_display = df_stat.copy()
            for c in ["media", "mediana", "min", "max"]:
                df_stat_display[c] = df_stat_display[c].apply(lambda v: fmt_eur(v) if not pd.isna(v) else "N/A")
            df_stat_display["variazione_%"] = df_stat_display["variazione_%"].apply(lambda v: fmt_pct(v) if v is not None else "N/A")
            st.dataframe(df_stat_display, hide_index=True, use_container_width=True)
        else:
            df_stat_display = df_stat.copy()
            df_stat_display["variazione_%"] = df_stat_display["variazione_%"].apply(lambda v: fmt_pct(v) if v is not None else "N/A")
            st.dataframe(df_stat_display, hide_index=True, use_container_width=True)
    with col_b:
        st.write("Sparkline")
        st.plotly_chart(spark, use_container_width=True)

    # Pulsanti rapidi intervallo
    st.markdown("**Intervalli rapidi**")
    cols = st.columns(6)
    for days, c in zip([2, 5, 7, 10, 14, 30], cols):
        with c:
            if st.button(f"{days}g", key=f"quick_{metric}_{days}"):
                end = st.session_state.get("last_date_selected") or date.today()
                start = end - timedelta(days=days - 1)
                st.session_state["date_quick_override"] = (start, end)
                st.experimental_rerun()


# =============================
# DEGRADAMENTO
# =============================

def degradamento_tab(df_range: pd.DataFrame):
    st.subheader("Degradamento")
    if df_range.empty:
        st.info("Nessun dato nel periodo.")
        return
    dfm = df_range.sort_values("data_dt")

    metrics_conf = [
        ("ctr_link", True, False, "CTR link (%)", "down", CONFIG["THRESHOLDS"]["ctr_link_var_pct_neg"]),
        ("cpm", False, True, "CPM (€)", "up", CONFIG["THRESHOLDS"]["cpm_var_pct_pos"]),
        ("frequenza", False, False, "Frequenza", "up_abs", CONFIG["THRESHOLDS"]["freq_increase_abs"]),
        ("cpl", False, True, "CPL (€)", "up", CONFIG["THRESHOLDS"]["cpl_var_pct_pos"]),
    ]

    rows = []
    for metric, is_pct, is_curr, label, direction, thr in metrics_conf:
        s = pd.to_numeric(dfm[metric], errors="coerce")
        first = s.iloc[0] if len(s)>0 else None
        last = s.iloc[-1] if len(s)>0 else None
        varp = variation_pct(first, last)
        if direction == "down":
            consec = consecutive_moves(s, "down")
            degrade = varp is not None and varp <= thr
        elif direction == "up":
            consec = consecutive_moves(s, "up")
            degrade = varp is not None and varp >= thr
        else:  # up_abs
            consec = consecutive_moves(s, "up")
            degrade = (last is not None and first is not None and (last - first) >= thr)
        rows.append({
            "metrica": label,
            "var_%": varp,
            "giorni_consecutivi": consec,
            "DEGRADO": "Sì" if degrade else "No",
        })

    out = pd.DataFrame(rows)
    # Ordinamento: prima quelli con DEGRADO sì e per severità var% (assoluto maggiore)
    out["sev_abs"] = out["var_%"].apply(lambda v: abs(v) if v is not None else -1)
    out = out.sort_values(by=["DEGRADO", "sev_abs"], ascending=[False, False]).drop(columns=["sev_abs"])

    # Colori riga
    def _row_style(row):
        if row["DEGRADO"] == "Sì":
            return ["background-color: #fee2e2"] * len(row)
        return [""] * len(row)

    # Formatta var%
    out_disp = out.copy()
    out_disp["var_%"] = out_disp["var_%"].apply(lambda v: fmt_pct(v) if v is not None else "N/A")
    st.dataframe(out_disp, hide_index=True, use_container_width=True)


# =============================
# SALUTE & AZIONI (RULE ENGINE)
# =============================
@dataclass
class Rule:
    id: str
    descrizione: str
    severita: str  # info | warning | critical
    azione_consigliata: str

    def attiva(self, ctx: Dict[str, pd.Series]) -> Tuple[bool, Optional[str]]:
        """Ritorna (attivata, prova_testuale)"""
        raise NotImplementedError


class RuleRefreshCreativo(Rule):
    def attiva(self, ctx: Dict[str, pd.Series]):
        ctr = ctx.get("ctr_link")
        cpm = ctx.get("cpm")
        freq = ctx.get("frequenza")
        cpl = ctx.get("cpl")
        if any(s is None or len(s.dropna()) < 2 for s in [ctr, cpm, freq, cpl]):
            return False, None
        ctr_consec_down = consecutive_moves(ctr, "down")
        cpm_up = (cpm.iloc[-1] - cpm.iloc[0]) > 0
        freq_up = (freq.iloc[-1] - freq.iloc[0]) > 0
        cpl_up = (cpl.iloc[-1] - cpl.iloc[0]) > 0
        cond = (ctr_consec_down >= 7) and cpm_up and freq_up and cpl_up
        prova = f"CTR link in calo per {ctr_consec_down}g, CPM↑, Frequenza↑, CPL↑" if cond else None
        return cond, prova


class RuleCpmUpCtrDown(Rule):
    def attiva(self, ctx: Dict[str, pd.Series]):
        ctr = ctx.get("ctr_link"); cpm = ctx.get("cpm")
        if any(s is None or len(s.dropna()) < 2 for s in [ctr, cpm]):
            return False, None
        ctr_var = variation_pct(ctr.iloc[0], ctr.iloc[-1])
        cpm_var = variation_pct(cpm.iloc[0], cpm.iloc[-1])
        cond = (ctr_var is not None and ctr_var < 0) and (cpm_var is not None and cpm_var > 0)
        prova = f"CTR link {fmt_pct(ctr_var)}, CPM {fmt_pct(cpm_var)}" if cond else None
        return cond, prova


class RuleFreqAlta(Rule):
    def attiva(self, ctx: Dict[str, pd.Series]):
        freq = ctx.get("frequenza")
        if freq is None or len(freq.dropna()) < 3:
            return False, None
        # Frequenza > 2 per >=3 giorni (anche non consecutivi)
        over2 = (freq > 2.0).sum()
        cond = over2 >= 3
        prova = f"Frequenza > 2 per {int(over2)} giorni" if cond else None
        return cond, prova


class RuleCplOverTarget(Rule):
    def __init__(self, id, descrizione, severita, azione_consigliata, cpl_target: Optional[float]):
        super().__init__(id, descrizione, severita, azione_consigliata)
        self.cpl_target = cpl_target

    def attiva(self, ctx: Dict[str, pd.Series]):
        if self.cpl_target is None:
            return False, None
        cpl = ctx.get("cpl")
        if cpl is None or cpl.dropna().empty:
            return False, None
        mean_cpl = cpl.mean()
        cond = mean_cpl > self.cpl_target
        prova = f"CPL medio {fmt_eur(mean_cpl)} > target {fmt_eur(self.cpl_target)}" if cond else None
        return cond, prova


def salute_azioni_tab(df_range: pd.DataFrame):
    st.subheader("Salute & Azioni")
    if df_range.empty:
        st.info("Nessun dato nel periodo.")
        return
    dfm = df_range.sort_values("data_dt")

    # Costruisci contesto serie
    ctx = {m: pd.to_numeric(dfm[m], errors="coerce") for m in ["ctr_link", "cpm", "frequenza", "cpl"]}

    rules: List[Rule] = [
        RuleRefreshCreativo(
            id="refresh_creativo",
            descrizione="CTR link ↓ per 7g e CPM/Frequenza/CPL in aumento",
            severita="critical",
            azione_consigliata="Refresh creativo (nuovi hook, variazioni di formato, rotazione).",
        ),
        RuleCpmUpCtrDown(
            id="cpm_up_ctr_down",
            descrizione="CPM in aumento e CTR link in calo",
            severita="warning",
            azione_consigliata="Rivedi targeting/placement e testa creatività brevi.",
        ),
        RuleFreqAlta(
            id="freq_alta",
            descrizione="Frequenza > 2 per ≥3 giorni",
            severita="warning",
            azione_consigliata="Amplia audience o ruota creatività.",
        ),
        RuleCplOverTarget(
            id="cpl_over_target",
            descrizione="CPL medio sopra target",
            severita="info",
            azione_consigliata="Allinea offerta/landing o lavora su CPM e CTR link.",
            cpl_target=st.session_state.cpl_target,
        ),
    ]

    attive = []
    for r in rules:
        on, prova = r.attiva(ctx)
        if on:
            attive.append((r, prova))

    # Stato salute aggregato
    health_color = CONFIG["HEALTH_COLORS"]["ok"]
    health_label = "Verde"
    if any(r.severita == "critical" for r, _ in attive):
        health_color = CONFIG["HEALTH_COLORS"]["crit"]
        health_label = "Rosso"
    elif any(r.severita == "warning" for r, _ in attive):
        health_color = CONFIG["HEALTH_COLORS"]["warn"]
        health_label = "Giallo"

    st.markdown(f"**Stato di salute**: <span style='color:{health_color};font-weight:600'>{health_label}</span>", unsafe_allow_html=True)

    if attive:
        for r, prova in attive:
            st.markdown(f"- **[{r.severita.upper()}] {r.descrizione}** → {r.azione_consigliata}. _Prova_: {prova}")
    else:
        st.success("Nessuna criticità rilevata.")

    st.markdown("---")
    st.markdown("### Proiezioni (prossimi 7 giorni)")

    def projection_plot(metric: str, title: str, is_pct=False, is_curr=False):
        s = pd.to_numeric(dfm[metric], errors="coerce").dropna()
        if len(s) < 3:
            st.info(f"{title}: N/A (servono almeno 3 punti)")
            return
        # regressione lineare su indice temporale
        X = np.arange(len(s)).reshape(-1, 1)
        y = s.values
        lr = LinearRegression().fit(X, y)
        y_pred = lr.predict(X)
        resid = y - y_pred
        std = np.std(resid)
        # forecast 7 giorni
        Xf = np.arange(len(s), len(s) + 7).reshape(-1, 1)
        yf = lr.predict(Xf)
        x_dates = list(dfm["data_dt"].dropna().iloc[-len(s):]) + [dfm["data_dt"].dropna().iloc[-1] + timedelta(days=i+1) for i in range(7)]
        y_all = np.concatenate([y, yf])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_dates[:len(s)], y=y, mode="lines+markers", name="Storico"))
        fig.add_trace(go.Scatter(x=x_dates, y=y_all, mode="lines", name="Trend"))
        # banda "fiducia" semplice ±1 std
        upper = np.concatenate([y_pred + std, yf + std])
        lower = np.concatenate([y_pred - std, yf - std])
        fig.add_trace(go.Scatter(x=x_dates, y=upper, mode="lines", name="+1σ", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=x_dates, y=lower, mode="lines", name="-1σ", line=dict(dash="dot"), fill="tonexty"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Data", yaxis_title=("%" if is_pct else ("€" if is_curr else "Valore")))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        projection_plot("ctr_link", "CTR link (%)", is_pct=True)
    with col2:
        projection_plot("cpm", "CPM (€)", is_curr=True)
    projection_plot("cpl", "CPL (€)", is_curr=True)

    # Stima ausiliaria CPL da CPM & CTR link
    st.markdown("#### Stima ausiliaria CPL")
    s_cpm = pd.to_numeric(dfm["cpm"], errors="coerce")
    s_ctrl = pd.to_numeric(dfm["ctr_link"], errors="coerce") / 100.0  # in decimale
    s_cpl = pd.to_numeric(dfm["cpl"], errors="coerce")
    valid = (~s_cpm.isna()) & (~s_ctrl.isna()) & (~s_cpl.isna()) & (s_ctrl > 0)
    if valid.sum() >= 5:
        est = s_cpm[valid] / (1000.0 * s_ctrl[valid])
        alpha = (s_cpl[valid] / est).mean()
        last_cpm = s_cpm.iloc[-1]
        last_ctrl = s_ctrl.iloc[-1]
        if last_ctrl > 0:
            cpl_aux = (last_cpm / (1000.0 * last_ctrl)) * alpha
            st.info(f"CPL stimato (euristico) ≈ {fmt_eur(cpl_aux)}  (α={alpha:.2f})")
        else:
            st.info("CPL stimato: N/A")
    else:
        st.info("CPL stimato: N/A (dati storici insufficienti)")


# =============================
# BUDGET & OBIETTIVI
# =============================

def budget_obiettivi_tab(df_range: pd.DataFrame):
    st.subheader("Budget & Obiettivi")
    if df_range.empty:
        st.info("Nessun dato nel periodo.")
        return

    dfm = df_range.sort_values("data_dt")
    cpl_mean = pd.to_numeric(dfm["cpl"], errors="coerce").mean()

    # Budget mensile disponibile
    # Usa valore di riga se presente, altrimenti globale
    if pd.to_numeric(dfm["budget_adv_disponibile_mese"], errors="coerce").notna().any():
        budget_mensile = pd.to_numeric(dfm["budget_adv_disponibile_mese"], errors="coerce").dropna().iloc[-1]
    else:
        budget_mensile = st.session_state.global_budget_mese

    lead_desiderati = None
    if pd.to_numeric(dfm["numero_lead_desiderati_mese"], errors="coerce").notna().any():
        lead_desiderati = int(pd.to_numeric(dfm["numero_lead_desiderati_mese"], errors="coerce").dropna().iloc[-1])
    else:
        lead_desiderati = st.session_state.global_lead_mese

    if cpl_mean is None or pd.isna(cpl_mean) or cpl_mean <= 0:
        st.warning("CPL medio non disponibile: impossibile calcolare obiettivi.")
        return

    # Lead/mese attesi = budget_mensile / CPL medio
    lead_attesi = None
    if budget_mensile is not None and budget_mensile > 0:
        lead_attesi = budget_mensile / cpl_mean

    # Budget giornaliero consigliato per centrare lead desiderati
    budget_consigliato = None
    if lead_desiderati and lead_desiderati > 0:
        budget_consigliato = lead_desiderati * cpl_mean / 30.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPL medio", fmt_eur(cpl_mean))
    with col2:
        st.metric("Lead/mese attesi", fmt_int(lead_attesi) if lead_attesi is not None else "N/A")
    with col3:
        st.metric("Budget giornaliero consigliato", fmt_eur(budget_consigliato) if budget_consigliato is not None else "N/A")

    # Gap e leve
    if lead_desiderati and lead_attesi is not None:
        gap = lead_desiderati - lead_attesi
        if gap > 0:
            st.warning(f"Gap vs obiettivo: mancano ~{fmt_int(gap)} lead/mese.")
            # suggerisci due leve: riduzione CPL o aumento budget
            cpl_target = cpl_mean * (lead_attesi / lead_desiderati)
            budget_necessario = lead_desiderati * cpl_mean
            st.markdown(f"- Riduci **CPL** a **{fmt_eur(cpl_target)}** _oppure_ aumenta **budget mensile** a **{fmt_eur(budget_necessario)}**.")
        else:
            st.success("Obiettivo lead raggiungibile alle condizioni attuali.")


# =============================
# MAIN APP
# =============================

def main():
    st.set_page_config(page_title=CONFIG["APP_TITLE"], layout="wide")
    init_session_state()

    st.title(CONFIG["APP_TITLE"])
    st.caption(f"Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} (locale)")

    df = ensure_schema(st.session_state.df)

    azienda, livello, entita, drange = sidebar_filters(df)

    # Override rapido dal bottone intervallo (tabs metrica)
    if "date_quick_override" in st.session_state:
        drange = st.session_state.pop("date_quick_override")

    st.session_state["last_date_selected"] = drange[1]

    # Header breadcrumb
    breadcrumb = f"**{azienda} › {livello} › {entita}**"
    st.markdown(breadcrumb)

    # Filtra dati per entità e range
    df_range = filter_df(df, azienda, livello, entita, drange)
    df_range = parse_dates(df_range)

    # KPI snapshot
    kpi_snapshot(df_range)

    st.markdown("---")

    # Editing se entità specifica (non "Tutte")
    if entita != "Tutte":
        df_entita = filter_df(df, azienda, livello, entita, (date.min, date.max))
        editing_block(df_entita, azienda, livello, entita)
    else:
        st.info("Seleziona un'entità specifica per modificare i dati.")

    st.markdown("---")

    # Tabs per metrica + Degradamento + Salute & Azioni + Budget
    tabs = st.tabs([
        "CPL", "CPC", "CPC link", "CTR all", "CTR link", "Frequenza", "Impression", "Copertura", "CPM",
        "Degradamento", "Salute & Azioni", "Budget & Obiettivi"
    ])

    # Metrica tabs
    with tabs[0]:
        metric_tab(df_range, "cpl", "CPL (€)", is_currency=True)
    with tabs[1]:
        metric_tab(df_range, "cpc", "CPC (€)", is_currency=True)
    with tabs[2]:
        metric_tab(df_range, "cpc_link", "CPC link (€)", is_currency=True)
    with tabs[3]:
        metric_tab(df_range, "ctr_all", "CTR all (%)", is_pct=True)
    with tabs[4]:
        metric_tab(df_range, "ctr_link", "CTR link (%)", is_pct=True)
    with tabs[5]:
        metric_tab(df_range, "frequenza", "Frequenza")
    with tabs[6]:
        metric_tab(df_range, "impression", "Impression")
    with tabs[7]:
        metric_tab(df_range, "copertura", "Copertura")
    with tabs[8]:
        metric_tab(df_range, "cpm", "CPM (€)", is_currency=True)
    with tabs[9]:
        degradamento_tab(df_range)
    with tabs[10]:
        salute_azioni_tab(df_range)
    with tabs[11]:
        budget_obiettivi_tab(df_range)


if __name__ == "__main__":
    main()
