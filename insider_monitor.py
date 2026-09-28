"""
INSIDER TRANSACTION MONITOR v2
Paste a Form 4 / PDMR notice -> parse ticker + insider -> pull historical
insider dealings (Yahoo, US-deep) within lookback -> forward returns NET of
the local benchmark at 1D/1W/1M/3M -> dark navy/gold client dashboard.
"""
import json, os, re
import numpy as np
import pandas as pd

# ================= CONFIG =================
LOOKBACK_YEARS = 3
MIN_PERSON_EVENTS = 3   # fewer than this by the named person -> expand to all insiders + disclaimer
HORIZONS = [("1 DAY", "1D", 1), ("1 WEEK", "1W", 5), ("1 MONTH", "1M", 21), ("3 MONTHS", "3M", 63)]
LOG_FILE = "insider_log.csv"
LOGO_FILE = 'logo.png'      # drop the firm logo file next to the script to embed it
FIRM_NAME = 'BERENBERG'     # text fallback shown when no logo file is present
REPO_SLUG = 'shankkurra/insider-monitor'              # e.g. 'yourname/insider-monitor' -> adds a working submit box to the page
PASTED_NOTICES: list = []      # raw text notices
MANUAL_ENTRIES: list = []      # dict(ticker=, person=, role=, direction=, date=, detail=)
# ==========================================

INDEX_MAP = {".L": ("^FTSE", "FTSE 100"), ".DE": ("^GDAXI", "DAX"), ".PA": ("^FCHI", "CAC 40"),
             ".AS": ("^AEX", "AEX"), ".MI": ("FTSEMIB.MI", "FTSE MIB"), ".SW": ("^SSMI", "SMI"),
             ".MC": ("^IBEX", "IBEX 35"), ".ST": ("^OMX", "OMX S30"), ".CO": ("^OMXC25", "OMX C25"),
             ".OL": ("OSEBX.OL", "OSEBX"), ".HE": ("^OMXH25", "OMX H25"), ".BR": ("^BFX", "BEL 20"),
             ".VI": ("^ATX", "ATX"), ".T": ("^N225", "Nikkei 225"), ".HK": ("^HSI", "Hang Seng"),
             ".AX": ("^AXJO", "ASX 200"), ".TO": ("^GSPTSE", "TSX")}
DEFAULT_INDEX = ("^GSPC", "S&P 500")
EXCHANGE_SUFFIX = {"NASDAQ": "", "NYSE": "", "AMEX": "", "NYSE MKT": "", "NYSE AMERICAN": "",
                   "LSE": ".L", "LON": ".L", "XETRA": ".DE", "FRA": ".DE", "ETR": ".DE",
                   "EPA": ".PA", "AMS": ".AS", "BIT": ".MI", "SWX": ".SW", "VTX": ".SW",
                   "BME": ".MC", "STO": ".ST", "TYO": ".T", "HKG": ".HK", "ASX": ".AX", "TSX": ".TO"}

def auto_index(ticker):
    for suf, pair in INDEX_MAP.items():
        if ticker.upper().endswith(suf):
            return pair
    return DEFAULT_INDEX

# ---------------- notice parsing ----------------
def parse_regex(text):
    out = {}
    m = re.search(r"\(\s*([A-Z ]+?)\s*:\s*([A-Z0-9.\-]+)\s*\)", text)
    if m:
        exch, tkr = m.group(1).strip(), m.group(2).strip()
        out["ticker"] = tkr + EXCHANGE_SUFFIX.get(exch, "")
    tl = text.lower()
    if any(w in tl for w in ("purchase", "bought", "acquir", "buys ")):
        out["direction"] = "BUY"
    elif any(w in tl for w in ("sold", "sale", "dispos", "sells ")):
        out["direction"] = "SELL"
    m = re.search(r"Form 4:\s*([^,]+?),", text) or re.search(r":\s*([A-Z][\w .&'-]+?(?:LP|LLC|Inc|Ltd)?),", text)
    if m:
        out["person"] = m.group(1).strip()
    dates = re.findall(r"([A-Z][a-z]+ \d{1,2},? \d{4})", text) + re.findall(r"(\d{4}-\d{2}-\d{2})", text)
    if dates:
        out["date"] = str(pd.to_datetime(dates[-1]).date())     # latest date = closest to disclosure
    return out

def parse_llm(text):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = ("Extract from this insider-dealing notice. If the person is named only partially "
              "(surname or first name only), use web search to resolve their FULL name exactly as it "
              "appears in official filings for this company, and verify it is the right individual "
              "(role must match the notice; beware family members sharing a surname). Reply ONLY JSON: "
              '{"ticker": "Yahoo Finance symbol with exchange suffix (.L LSE, .DE Xetra, .PA Paris, none for US)", '
              '"person": "full name", "role": string|null, "direction": "BUY"|"SELL", '
              '"date": "YYYY-MM-DD latest transaction/disclosure date", "detail": "<=12 words"}'
              f"\n\nNotice:\n{text}")
    for tools in ([{"type": "web_search_20250305", "name": "web_search"}], None):
        try:
            kw = dict(model="claude-haiku-4-5-20251001", max_tokens=500, temperature=0,
                      messages=[{"role": "user", "content": prompt}])
            if tools:
                kw["tools"] = tools
            msg = client.messages.create(**kw)
            t = "".join(b.text for b in msg.content if b.type == "text")
            j = json.loads(t[t.index("{"): t.rindex("}") + 1])
            return {k: v for k, v in j.items() if v}
        except Exception:
            continue
    return None

def parse_notice(text):
    out = parse_llm(text) or {}
    rx = parse_regex(text)
    for k, v in rx.items():
        out.setdefault(k, v)
    out.setdefault("role", "")
    out.setdefault("detail", text.strip().replace("\n", " ")[:110])
    missing = [k for k in ("ticker", "person", "direction", "date") if k not in out]
    return (None, missing) if missing else (out, [])

# ---------------- data ----------------
_PX = {}
def closes(symbol):
    if symbol not in _PX:
        import yfinance as yf
        df = yf.download(symbol, period=f"{LOOKBACK_YEARS + 1}y", auto_adjust=True, progress=False)
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        _PX[symbol] = s.dropna()
    return _PX[symbol]

_NAMES = {}
def company_name(ticker):
    if ticker not in _NAMES:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            _NAMES[ticker] = info.get("shortName") or info.get("longName") or ""
        except Exception:
            _NAMES[ticker] = ""
    return _NAMES[ticker]

def yahoo_history(ticker, direction):
    """Historical insider dealings from Yahoo (deep for US, thin for Europe)."""
    try:
        import yfinance as yf
        it = yf.Ticker(ticker).insider_transactions
        if it is None or not len(it):
            return []
    except Exception:
        return []
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)
    out = []
    for _, r in it.iterrows():
        txt = f'{r.get("Text", "")} {r.get("Transaction", "")}'.lower()
        d = "BUY" if "purchase" in txt or "buy" in txt else "SELL" if "sale" in txt or "sold" in txt else None
        dt = pd.to_datetime(r.get("Start Date"), errors="coerce")
        if d != direction or pd.isna(dt) or dt < cutoff:
            continue
        out.append(dict(ticker=ticker, person=str(r.get("Insider", "")).title(),
                        role=str(r.get("Position", "")), direction=d, date=str(dt.date()),
                        detail=str(r.get("Text", "")).strip(), source="yahoo"))
    return out

def norm_key(e):
    p = re.sub(r"[^a-z0-9]", "", str(e["person"]).lower())
    return (e["ticker"].upper(), str(e["date"]), e["direction"], p)

def fwd_net(ticker, date):
    """Per-horizon (net_return, complete) vs auto benchmark, from disclosure-day close."""
    isym, iname = auto_index(ticker)
    try:
        s, ix = closes(ticker.upper()), closes(isym)
    except Exception:
        return None
    d = pd.Timestamp(date)
    si = s.index[s.index >= d]
    xi = ix.index[ix.index >= d]
    if not len(si) or not len(xi):
        return None
    i0, j0 = s.index.get_loc(si[0]), ix.index.get_loc(xi[0])
    e_s, e_x = float(s.iloc[i0]), float(ix.iloc[j0])
    rows = {}
    for _, code, n in HORIZONS:
        ok = (i0 + n < len(s)) and (j0 + n < len(ix))
        rs = float(s.iloc[min(i0 + n, len(s) - 1)]) / e_s - 1
        rx = float(ix.iloc[min(j0 + n, len(ix) - 1)]) / e_x - 1
        rows[code] = {"stock": rs, "index": rx, "net": rs - rx, "complete": ok}
    return {"index_name": iname, "rows": rows}

# ---------------- html ----------------
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');
:root{--bg:#070b14;--panel:#0c1424;--tile:#0e182e;--line:#1d2a45;--gold:#c9a45f;
--ink:#e9e5da;--dim:#7d89a1;--dim2:#b3bccc;--grn:#7ec9a1;--red:#d1706b}
body{background:var(--bg);color:var(--ink);font-family:'Playfair Display',Georgia,serif;margin:0;padding:40px 22px}
.wrap{max-width:980px;margin:0 auto}
.layout{display:flex;gap:28px;align-items:flex-start}
.tabs{width:130px;flex-shrink:0;position:sticky;top:32px}
.tabbtn{display:block;width:100%;text-align:left;background:transparent;border:none;border-left:2px solid var(--line);
color:var(--dim);font-family:'Playfair Display',Georgia,serif;font-size:11px;font-weight:700;letter-spacing:.24em;
padding:13px 14px;cursor:pointer}
.tabbtn.on{color:var(--gold);border-left-color:var(--gold)}
.tabpane{display:none}
.tabpane.on{display:block}
.main{flex:1;min-width:0}
@media(max-width:640px){.layout{flex-direction:column}.tabs{position:static;width:100%;display:flex}
.tabbtn{border-left:none;border-bottom:2px solid var(--line);text-align:center}.tabbtn.on{border-bottom-color:var(--gold)}}
.mast{display:flex;justify-content:space-between;align-items:center;margin-bottom:34px}
h1{font-family:'Playfair Display',Georgia,serif;font-size:26px;font-weight:700;letter-spacing:.02em;margin:0}
.stamp{width:74px;height:74px;border:1px solid var(--gold);border-radius:50%;display:flex;align-items:center;
justify-content:center;text-align:center;color:var(--gold);font-size:8px;letter-spacing:.22em;line-height:1.8;flex-shrink:0}
.sec{color:var(--gold);font-size:11px;font-weight:600;letter-spacing:.28em;margin:38px 0 14px}
.panel{background:var(--panel);border:1px solid var(--line);padding:24px 26px;margin-bottom:8px}
.tkr{font-family:'Playfair Display',Georgia,serif;font-size:34px;font-weight:700;margin:0 0 2px}
.note{color:var(--gold);font-size:12px;font-style:italic;margin:-8px 0 16px}
.sub{color:var(--dim2);font-size:12.5px;letter-spacing:.04em;margin-bottom:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);margin-bottom:26px}
.tile{background:var(--tile);padding:15px 16px}
.tl{color:var(--dim);font-size:10px;letter-spacing:.22em;margin-bottom:8px}
.tv{font-family:'Playfair Display',Georgia,serif;font-size:27px;font-weight:600;margin-bottom:9px}
.tm{color:var(--dim);font-size:10.5px;line-height:1.75}.tm b{color:var(--ink);font-weight:600}
.tblhead{color:var(--dim);font-size:10px;letter-spacing:.24em;margin:6px 0 10px}
table{width:100%;border-collapse:collapse;font-size:12px;font-family:'Playfair Display',Georgia,serif}
th{color:var(--dim);font-size:10px;letter-spacing:.14em;text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th.r,td.r{text-align:right}
td{padding:11px 10px;border-bottom:1px solid #131d33;vertical-align:top;color:var(--ink)}
.pos{color:var(--grn)}.neg{color:var(--red)}.mut{color:var(--dim)}
.badge{border:1px solid var(--gold);color:var(--gold);font-size:9px;letter-spacing:.2em;padding:3px 9px;margin-left:12px;vertical-align:middle}
textarea{width:100%;box-sizing:border-box;background:#0a1120;border:1px solid var(--line);color:var(--ink);
font-family:'Playfair Display',Georgia,serif;font-size:13px;padding:12px 14px;resize:vertical;outline:none}
textarea:focus{border-color:var(--gold)}
button.gold{background:var(--gold);color:#0a0f1c;border:none;font-family:'Playfair Display',Georgia,serif;
font-weight:700;font-size:12px;letter-spacing:.18em;padding:10px 26px;cursor:pointer}
button.gold:hover{opacity:.88}
.foot{color:#55617a;font-size:10px;line-height:1.8;margin-top:34px;border-top:1px solid var(--line);padding-top:14px}
i{color:var(--dim)}
.matrix td{font-size:14px;padding:13px 10px;background:#0e182e}
.matrix tr.spacer td{padding:9px 0;border-bottom:none;background:transparent}
.matrix tr.winrow td{color:var(--ink);font-size:14px;padding:13px 10px;border-bottom:none;background:#0e182e}
.matrix tr.winrow td.r{color:var(--ink);font-weight:600}
.brand{color:var(--gold);font-family:'Playfair Display',Georgia,serif;font-size:13px;font-weight:700;letter-spacing:.55em;margin-bottom:12px}
.matrix tr.netrow td{border-top:2px solid var(--gold);background:#0e182e;font-weight:700;font-size:14px}
.matrix th.r{text-align:right}
.winline{color:var(--dim2);font-size:12px;margin-top:12px;letter-spacing:.02em}
.winline b{color:var(--ink)}
</style>"""

def pfmt(v, cls=True):
    c = "pos" if v > 0 else "neg" if v < 0 else "mut"
    s = f"{v:+.2%}"
    return f'<span class="{c}">{s}</span>' if cls else s

def ticker_block(ticker, focal, events, perf):
    """One ticker section: TICKER - Name, then the 3-row matrix: stock / index / net."""
    iname = next((perf[norm_key(e)]["index_name"] for e in events if norm_key(e) in perf), "index")
    name = company_name(ticker)
    title = f"{ticker} &mdash; {name}" if name else ticker

    stats, wins = {}, []
    for _, code, _ in HORIZONS:
        done = [perf[norm_key(e)]["rows"][code] for e in events if norm_key(e) in perf]
        done = [r for r in done if r["complete"]]
        if done:
            nets = np.array([r["net"] for r in done])
            hit = float(np.mean(nets > 0)) if focal["direction"] == "BUY" else float(np.mean(nets < 0))
            stats[code] = {"stock": float(np.mean([r["stock"] for r in done])),
                           "index": float(np.mean([r["index"] for r in done])),
                           "net": float(nets.mean()), "n": len(done)}
            wins.append(f'<td class="r">{hit:.0%}</td>')
        else:
            stats[code] = None
            wins.append('<td class="r mut">&mdash;</td>')

    def row(label, key, cls=""):
        cells = ""
        for _, code, _ in HORIZONS:
            s = stats[code]
            cells += (f'<td class="r">{pfmt(s[key])}</td>' if s else
                      '<td class="r mut">&mdash;</td>')
        return f'<tr class="{cls}"><td>{label}</td>{cells}</tr>'

    side = "purchases" if focal["direction"] == "BUY" else "disposals"
    perf_word = "outperformed" if focal["direction"] == "BUY" else "underperformed"
    who = focal.get("_scope_person")
    scope = f"historical {side} by {who}" if who else f"historical insider {side}"
    whose = f"{who}&rsquo;s historical {side}" if who else f"historical insider {side} in {ticker}"
    note = focal.get("_note")
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f"""<div class="panel">
<div class="tkr">{title} <span class="badge">{focal['direction']}</span></div>
<div class="sub">Average forward performance across {scope} &middot; lookback {LOOKBACK_YEARS}y</div>
{note_html}
<table class="matrix"><tr><th></th>{''.join(f'<th class="r">{l}</th>' for l, _, _ in HORIZONS)}</tr>
{row(ticker, "stock")}
{row(iname, "index")}
{row("PERFORMANCE VS INDEX", "net", "netrow")}
<tr class="spacer"><td colspan="{len(HORIZONS) + 1}"></td></tr>
<tr class="winrow"><td>WIN RATE VS {iname.upper()}</td>{''.join(wins)}</tr></table>
<div class="winline">Win rate: out of all {whose} in the sample, the share that {perf_word} the {iname} over each window.</div>
</div>"""

def aggregates_block(all_events, perf, num="I"):
    rows = ""
    for side, word in (("BUY", "BUYS"), ("SELL", "SELLS")):
        ev = [e for e in all_events if e["direction"] == side and norm_key(e) in perf]
        if not ev:
            continue
        cells = ""
        for _, code, _ in HORIZONS:
            done = [perf[norm_key(e)]["rows"][code] for e in ev]
            done = [r for r in done if r["complete"]]
            if done:
                arr = np.array([r["net"] for r in done])
                hit = np.mean(arr > 0) if side == "BUY" else np.mean(arr < 0)
                good = arr.mean() > 0 if side == "BUY" else arr.mean() < 0
                cells += (f'<td class="r"><span class="{"pos" if good else "neg"}">{arr.mean():+.2%}</span>'
                          f'<br><span class="mut">hit {hit:.0%} &middot; n={len(done)}</span></td>')
            else:
                cells += '<td class="r mut">&mdash;</td>'
        rows += f'<tr><td>{word} <span class="mut">({len(ev)} events)</span></td><td class="mut"></td>{cells}</tr>'
    if not rows:
        return ""
    return f"""<div class="sec">{num}. SIGNAL AGGREGATES &middot; ALL LOGGED NAMES</div><div class="panel">
<table><tr><th>DIRECTION</th><th></th>{''.join(f'<th class="r">{c}</th>' for _, c, _ in HORIZONS)}</tr>{rows}</table>
<div class="tm" style="margin-top:10px;color:var(--dim)">Mean net-of-benchmark forward return across every event below.
Hit = net &gt; 0 for buys, net &lt; 0 for sells. Completed horizons only.</div></div>"""

def logo_html():
    if os.path.exists(LOGO_FILE):
        import base64
        ext = LOGO_FILE.rsplit(".", 1)[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml"}.get(ext, "image/png")
        b64 = base64.b64encode(open(LOGO_FILE, "rb").read()).decode()
        return f'<img src="data:{mime};base64,{b64}" style="height:30px;display:block;margin-bottom:14px">'
    return f'<div class="brand">{FIRM_NAME}</div>' if FIRM_NAME else ""

def build_dashboard(blocks_data, all_events, perf):
    blocks = sorted(blocks_data, key=lambda b: str(b[1].get("date", "")), reverse=True)
    latest_html = ticker_block(*blocks[0], perf) if blocks else '<div class="sub">No evaluations yet.</div>'
    history_html = "".join(ticker_block(t, f, ev, perf) for t, f, ev in blocks)
    brand = logo_html()
    submit = ""
    if REPO_SLUG:
        submit = (
            '<div class="sec">I. SUBMIT THE NOTICE</div><div class="panel">'
            '<div class="sub" style="margin-bottom:10px">PASTED NOTIFICATION</div>'
            '<textarea id="nb" rows="3" placeholder="Paste the Form 4 / PDMR headline here..."></textarea>'
            '<div style="margin-top:12px"><button class="gold" onclick="submitNotice()">EVALUATE</button>'
            '<span class="sub" id="status" style="margin-left:14px"></span></div></div>'
            "<script>function submitNotice(){var t=document.getElementById('nb').value.trim();"
            "if(!t){alert('Paste a notice first');return;}"
            "window.open('https://github.com/" + REPO_SLUG +
            "/issues/new?title='+encodeURIComponent('NOTICE: '+t.slice(0,60))+'&body='+encodeURIComponent(t),'_blank');"
            "document.getElementById('status').textContent="
            "'Evaluating \u2014 the result appears below in about 2 minutes...';"
            "setInterval(function(){location.reload();},30000);}</script>")
    findings_num = "II" if REPO_SLUG else "I"
    ts = pd.Timestamp.today().strftime("%d %B %Y").upper()
    tail = """<script>
function showTab(k){['e','h'].forEach(function(x){
 document.getElementById('tab-'+x).classList.toggle('on',x===k);
 document.getElementById('tb-'+x).classList.toggle('on',x===k);});}
var M=['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER'];
var D=new Date();
document.getElementById('dt').textContent=D.getDate()+' '+M[D.getMonth()]+' '+D.getFullYear();
</script>"""
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Insider Transaction Monitor</title>{CSS}</head>
<body><div class="wrap">
<div class="mast"><div>{brand}<h1>Insider Transaction Monitor</h1>
<div class="sub" style="margin:6px 0 0">DIRECTOR &amp; PDMR DEALINGS &middot; <span id="dt">{ts}</span></div></div>
<div class="stamp">FORWARD<br>RETURNS</div></div>
<div class="layout">
<div class="tabs">
<button class="tabbtn on" id="tb-e" onclick="showTab('e')">EVALUATE</button>
<button class="tabbtn" id="tb-h" onclick="showTab('h')">HISTORY</button>
</div>
<div class="main">
<div class="tabpane on" id="tab-e">
{submit}
<div class="sec">{findings_num}. FINDINGS &middot; LATEST</div>
{latest_html}
</div>
<div class="tabpane" id="tab-h">
{aggregates_block(all_events, perf, "I")}
<div class="sec">II. ALL LOGGED NAMES</div>
{history_html}
</div>
</div>
</div>
<div class="foot">Returns measured from the close of the first trading day on/after the event date; horizons of 1/5/21/63
trading days; net = stock return minus local benchmark over the identical window. Historical events sourced from Yahoo
Finance insider filings (transaction dates; disclosure may lag up to two business days). &mdash; marks horizons not yet
complete; they are excluded from all averages. Internal analytical tool &mdash; not investment advice or a recommendation.</div>
</div>{tail}</body></html>"""
    open("insider_dashboard.html", "w").write(html)

def load_log():
    cols = ["ticker", "person", "role", "direction", "date", "detail"]
    return pd.read_csv(LOG_FILE).reindex(columns=cols) if os.path.exists(LOG_FILE) else pd.DataFrame(columns=cols)

def run():
    log = load_log()
    focal_new = []
    for txt in PASTED_NOTICES:
        e, missing = parse_notice(txt)
        if e is None:
            print(f"  ! could not parse notice (missing {missing}) — add it via MANUAL_ENTRIES:\n    {txt[:90]}")
            continue
        focal_new.append(e)
    focal_new += [{**{"role": "", "detail": ""}, **m} for m in MANUAL_ENTRIES]
    for e in focal_new:
        if not len(log) or not (log.apply(lambda r: norm_key(r) == norm_key(e), axis=1)).any():
            log = pd.concat([log, pd.DataFrame([{k: e.get(k, "") for k in
                            ("ticker", "person", "role", "direction", "date", "detail")}])], ignore_index=True)
    if not len(log):
        print("Nothing to evaluate — paste notices into PASTED_NOTICES or add MANUAL_ENTRIES.")
        return
    log.to_csv(LOG_FILE, index=False)

    blocks, all_events, perf, seen = [], [], {}, set()
    for (tkr,), grp in log.groupby(["ticker"]):
        focal = grp.sort_values("date").iloc[-1].to_dict()
        events = [r.to_dict() | {"source": "log"} for _, r in grp.iterrows()
                  if r["direction"] == focal["direction"]]
        for h in yahoo_history(tkr.upper(), focal["direction"]):
            if norm_key(h) not in {norm_key(e) for e in events}:
                events.append(h)
        _stop = {"lp", "llc", "inc", "ltd", "plc", "ag", "se", "jr", "sr", "mr", "mrs", "dr", "the"}
        def _toks(name):
            return {w for w in re.findall(r"[a-z0-9]+", str(name).lower()) if len(w) > 1 and w not in _stop}
        pt = _toks(focal.get("person", ""))
        def _match(e):
            q = _toks(e.get("person", ""))
            if not pt or not q:
                return False
            need = min(2, len(pt), len(q))
            return len(pt & q) >= need
        pev = [e for e in events if _match(e)]
        word = "purchases" if focal["direction"] == "BUY" else "disposals"
        if len(pev) >= MIN_PERSON_EVENTS:
            events = pev
            focal["_scope_person"] = str(focal["person"])
            focal["_note"] = None
        else:
            focal["_scope_person"] = None
            n_p = len(pev)
            if pt and len(events) > n_p:
                have = (f"only {n_p} recorded {word[:-1] if n_p == 1 else word}" if n_p
                        else f"no other recorded {word}")
                focal["_note"] = (f"Note: {focal['person']} has {have} in the lookback window, "
                                  f"so the table aggregates all insider {word} in the company.")
            else:
                focal["_note"] = None
        for e in events:
            k = norm_key(e)
            if k not in seen:
                seen.add(k)
                p = fwd_net(e["ticker"], e["date"])
                if p:
                    perf[k] = p
        events = [e for e in events if norm_key(e) in perf]
        if events:
            blocks.append((tkr.upper(), focal, events))
            all_events += events
        else:
            print(f"  ! no price data for {tkr} — check ticker suffix")
    build_dashboard(blocks, all_events, perf)
    print(f"Dashboard built: {len(blocks)} names, {len(all_events)} events -> insider_dashboard.html | log: {LOG_FILE}")

def _main():
    import shutil
    body = re.sub(r"\s+", " ", os.environ.get("NOTICE_BODY", "")).strip()
    if body:
        with open("notices.txt", "a") as f:
            f.write("\n" + body + "\n")
    if os.path.exists("notices.txt"):
        PASTED_NOTICES[:] = [ln.strip() for ln in open("notices.txt")
                             if ln.strip() and not ln.strip().startswith("#")]
    if os.path.exists("manual_entries.csv"):
        df = pd.read_csv("manual_entries.csv").fillna("")
        MANUAL_ENTRIES[:] = [r.to_dict() for _, r in df.iterrows() if str(r.get("ticker", "")).strip()]
    run()
    if os.path.exists("insider_dashboard.html"):
        os.makedirs("docs", exist_ok=True)
        shutil.copy("insider_dashboard.html", "docs/index.html")
    open("notices.txt", "w").write(
        "# PASTE NOTICES BELOW THIS LINE - one per line, then commit.\n"
        "# The dashboard rebuilds automatically (~2 min) and this file resets itself.\n")

if __name__ == "__main__":
    _main()
