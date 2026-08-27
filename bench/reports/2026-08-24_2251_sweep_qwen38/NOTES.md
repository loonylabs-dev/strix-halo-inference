# Adjudikation zum Screening-Lauf (24.08.2026, spät)

Die rows.jsonl/summary.json bleiben unverändert — das hier sind die
Korrekturen der Interpretation, nicht der Messung.

- **laguna · prose-cache: FAIL → PASS (Checker-Artefakt).** Die Antwort
  (im laguna-recheck mit `answer` gespeichert) erklärt Prefix-Caching
  einwandfrei, schreibt aber „Prefix-Caching"/„gecacht" — der wörtliche
  Keyword-Match auf „cache" griff nicht. Checker seit diesem Lauf auf
  Wortstämme umgestellt (tasklib.py); Neubewertung der gespeicherten
  Antwort: PASS. Effektive Laguna-Wertung: **8/9**.
- **laguna · longctx-retrieval: FAIL bestätigt (reproduziert).** Zweimal
  gemessen (301,7 s und 292,0 s), beide Male 4.096 Tokens vollständig im
  Thinking verbrannt, kein ANSWER. Qwen (rocm-medium-spec) löst dieselbe
  Aufgabe korrekt in 230 s / 2.743 Tokens.
- **rocm-none-spec: 0/9 ist kein Modellergebnis.** Das Chat-Template
  kennt nur reasoning_effort xhigh/medium/low; „none" wirft eine
  Jinja-Exception. Ersetzt durch `rocm-nothink-spec`
  (enable_thinking:false), separat nachgemessen.
- **Probes der Qwen-Zellen fehlen in diesem Lauf** (System-Message-
  Remap kam erst danach); Task-Timings sind davon unberührt. pp/tg der
  Finalisten werden per `--probes-only`-Nachlauf ergänzt.
- **Leistungsprofil war durchgehend `balanced`** (siehe context.json) —
  bewusst, wegen Vergleichbarkeit mit allen bisherigen docs/-Zahlen.
