"""Render one run's trace as a self-contained HTML page."""
from __future__ import annotations

import html
import time

from .store import Store

CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e3e1da;--ok:#1a7f4b;--bad:#b3261e;
--warn:#9a6700;--info:#1f5f8b}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e22;--ink:#ececee;--mute:#9a9aa2;
--line:#2e2e34;--ok:#5fd08f;--bad:#ff8a80;--warn:#e3b341;--info:#7fb8e6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:32px 0 10px;text-transform:uppercase;
letter-spacing:.06em;color:var(--mute)}
.sub{color:var(--mute);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:20px}.stat span{color:var(--mute);font-size:12px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;white-space:nowrap}tr:last-child td{border-bottom:0}
code{font:12px ui-monospace,Menlo,monospace}
.k{font-weight:600;white-space:nowrap}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.info{color:var(--info)}.mute{color:var(--mute)}
"""
KIND_CLASS = {"step_ok": "ok", "step_failed": "bad", "tool_retry": "warn", "decision": "info",
              "step_replayed": "mute", "permission_required": "warn", "permission_granted": "ok",
              "feedback": "info", "run_start": "", "run_end": ""}


def _e(x) -> str:
    return html.escape(str(x))


def render(store: Store, run_id: str) -> str:
    run, events, steps = store.get_run(run_id), store.events(run_id), store.steps(run_id)
    by_key = {s["key"]: s for s in steps}
    count = lambda kind: sum(e["kind"] == kind for e in events)  # noqa: E731
    stats = [
        (run["status"].replace("_", " "), "status"), (f"${run['cost_usd']:.2f}", "cost"),
        (len(steps), "steps executed"), (count("step_replayed"), "steps replayed from checkpoint"),
        (count("tool_retry"), "provider retries"),
        (sum(e["kind"] == "decision" and e["data"].get("action") == "regenerate" for e in events),
         "clips rejected by evals"),
        (count("run_start"), "process starts"),
    ]
    plan = by_key.get("plan", {}).get("output", {})
    shot_rows = []
    for shot in plan.get("shots", []):
        gens = [s for s in steps if s["key"].startswith(f"gen:{shot['id']}:")]
        for g in gens:
            tag = g["key"][4:]
            ev = by_key.get(f"eval:{tag}", {}).get("output")
            if not g["output"]["ok"]:
                verdict = f'<span class="bad">provider error</span> {_e(g["output"]["error"])}'
            elif ev is None:
                verdict = '<span class="mute">not evaluated</span>'
            elif ev["passed"]:
                scores = (ev.get("judge") or {}).get("scores")
                verdict = '<span class="ok">passed</span>' + (f" {_e(scores)}" if scores else "")
            else:
                bad = [f'{c["name"]} = {c["value"]} (want {c["threshold"]})' for c in ev["checks"]
                       if not c["passed"]]
                judge = ev.get("judge")
                if judge:  # the reviewer's tags, scores and one-line reason
                    bad.append(f'judge: {", ".join(judge["issues"])} {judge["scores"]} "{judge["note"]}"')
                verdict = f'<span class="bad">rejected</span> {_e("; ".join(bad))}'
            provider = g["output"].get("provider", "")
            shot_rows.append(f"<tr><td class=k>{_e(shot['id'])} {_e(shot['beat'])}</td><td><code>{_e(tag)}</code>"
                             f"</td><td>{_e(provider)}</td><td>${g['cost']:.2f}</td><td>{verdict}</td></tr>")
    t0 = events[0]["ts"] if events else time.time()
    event_rows = []
    for e in events:
        detail = ", ".join(f"{k}={v}" for k, v in e["data"].items() if k != "tool")
        event_rows.append(
            f"<tr><td class=mute>+{e['ts'] - t0:.2f}s</td><td class='k {KIND_CLASS.get(e['kind'], '')}'>"
            f"{_e(e['kind'])}</td><td><code>{_e(e['key'])}</code></td><td>{_e(e['data'].get('tool', ''))}</td>"
            f"<td>{_e(detail)}</td></tr>")
    stat_html = "".join(f"<div class=stat><b>{_e(v)}</b><span>{_e(k)}</span></div>" for v, k in stats)
    style = plan.get("style", {})
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Run {_e(run_id)} trace</title>
<style>{CSS}</style></head><body><main>
<h1>{_e(run['brief'])}</h1>
<p class=sub>Run <code>{_e(run_id)}</code> · config {_e(run['config']['name'])} · mood {_e(style.get('mood', '?'))}
 · palette {_e(' '.join(style.get('palette', [])))} · scopes granted: {_e(', '.join(run['state']['granted']))}</p>
<div class=stats>{stat_html}</div>
<h2>Shots and generation attempts</h2>
<div class=wrap><table><tr><th>Shot</th><th>Attempt</th><th>Provider</th><th>Cost</th><th>Outcome</th></tr>
{''.join(shot_rows)}</table></div>
<h2>Event timeline</h2>
<div class=wrap><table><tr><th>Time</th><th>Event</th><th>Step</th><th>Tool</th><th>Detail</th></tr>
{''.join(event_rows)}</table></div>
</main></body></html>"""
