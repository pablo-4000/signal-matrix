"""
run_report.py  —  marketintell2000
Calls the Anthropic API with web search, generates a market intelligence
report, and saves it as report.html in the repo root.
"""

import os
import sys
import anthropic
from datetime import datetime, timezone


# ── TICKERS ───────────────────────────────────────────────────────────────────
TICKERS = [
    "S&P 500 (SPY)",
    "Nasdaq 100 (QQQ)",
    "Gold (GLD / XAU/USD)",
    "Bitcoin (BTC-USD)",
    "Crude Oil (WTI)",
    "US Treasuries (10-year yield)",
]

# ── PROMPT ────────────────────────────────────────────────────────────────────
def build_prompt():
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    tickers_str = "\n".join("- " + t for t in TICKERS)
    return (
        "Today is " + today + ".\n\n"
        "You are a professional market analyst. Search the web for current data and create "
        "a comprehensive market intelligence report for these assets:\n\n"
        + tickers_str + "\n\n"
        "For EACH asset provide exactly this structure:\n\n"
        "## [Asset Name]\n\n"
        "### Prices & Changes\n"
        "Search for the latest price. Provide a small table with: current price, "
        "1-day change (%), 1-week change (%), 1-month change (%).\n\n"
        "### Technical Signals\n"
        "Based on current price action:\n"
        "- RSI (14): value + signal (Overbought / Neutral / Oversold)\n"
        "- MACD: signal (Bullish / Bearish / Neutral)\n"
        "- Moving Averages (50/200): Above / Mixed / Below\n"
        "- Overall: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL\n\n"
        "### Latest News (last 48 hours)\n"
        "3-5 most important news items with brief price-impact notes.\n\n"
        "### Upcoming Events (next 14 days)\n"
        "Relevant economic releases, Fed meetings, earnings, options expiry, etc. "
        "Write 'No major events scheduled.' if none.\n\n"
        "### Key Price Drivers\n"
        "2-3 bullet points on macro or sector factors driving this asset.\n\n"
        "---\n\n"
        "After all assets, add a **Market Summary** section (3-5 sentences) covering "
        "the overall environment, dominant themes, and key risks.\n\n"
        "Use Markdown formatting. Be specific — search for actual current prices and data."
    )


# ── API CALL ──────────────────────────────────────────────────────────────────
def run_research(client):
    prompt = build_prompt()
    messages = [{"role": "user", "content": prompt}]

    print("  Calling Anthropic API...")

    # For server-side tools (web_search_20250305) the API handles search
    # execution internally. We loop only if the model needs multiple turns.
    for turn in range(10):  # safety cap
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

        print("  turn=" + str(turn) + " stop_reason=" + str(response.stop_reason))

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            # Append assistant message then feed back empty tool results
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [],
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            # Unknown stop reason — stop looping
            break

    # Extract text from the final response
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    return text.strip()


# ── MARKDOWN → HTML ───────────────────────────────────────────────────────────
def md_to_html(md):
    import re

    lines = md.split("\n")
    out = []
    in_ul = False
    in_ol = False
    in_table = False
    in_pre = False
    thead_done = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_table():
        nonlocal in_table, thead_done
        if in_table:
            out.append("</tbody></table>")
            in_table = False
            thead_done = False

    def inline(t):
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"__(.+?)__", r"<strong>\1</strong>", t)
        t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        # Colour trading signals
        for sig, cls in [
            ("STRONG BUY", "sig-sb"), ("STRONG SELL", "sig-ss"),
            ("BUY", "sig-b"), ("SELL", "sig-s"), ("NEUTRAL", "sig-n"),
        ]:
            t = t.replace(sig, '<span class="' + cls + '">' + sig + "</span>")
        # Colour +/- percentages
        t = re.sub(r"(\+[\d.]+\s*%)", r'<span class="pos">\1</span>', t)
        t = re.sub(r"(-[\d.]+\s*%)", r'<span class="neg">\1</span>', t)
        return t

    for line in lines:
        # Pre / code blocks
        if line.startswith("```"):
            close_lists()
            close_table()
            if not in_pre:
                out.append("<pre><code>")
                in_pre = True
            else:
                out.append("</code></pre>")
                in_pre = False
            continue
        if in_pre:
            out.append(line)
            continue

        # HR
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line.strip()):
            close_lists()
            close_table()
            out.append("<hr>")
            continue

        # Table rows
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                close_lists()
                out.append('<table>')
                in_table = True
                thead_done = False
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                # separator row — open tbody
                out.append("<tbody>")
                thead_done = True
            elif not thead_done:
                out.append(
                    "<thead><tr>"
                    + "".join("<th>" + inline(c) + "</th>" for c in cells)
                    + "</tr></thead>"
                )
            else:
                out.append(
                    "<tr>"
                    + "".join("<td>" + inline(c) + "</td>" for c in cells)
                    + "</tr>"
                )
            continue
        else:
            close_table()

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            close_lists()
            lvl = len(m.group(1))
            tag = "h2" if lvl <= 2 else "h3"
            out.append("<" + tag + ">" + inline(m.group(2)) + "</" + tag + ">")
            continue

        # Unordered list
        m = re.match(r"^[-*+]\s+(.*)", line)
        if m:
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append("<li>" + inline(m.group(1)) + "</li>")
            continue

        # Ordered list
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append("<li>" + inline(m.group(1)) + "</li>")
            continue

        # Blank line
        if not line.strip():
            close_lists()
            continue

        # Paragraph
        close_lists()
        out.append("<p>" + inline(line) + "</p>")

    close_lists()
    close_table()
    if in_pre:
        out.append("</code></pre>")

    return "\n".join(out)


# ── HTML PAGE BUILDER ─────────────────────────────────────────────────────────
def build_page(body_html, timestamp):
    # Build page using string concatenation — no .format(), no template engine,
    # so CSS curly braces never cause KeyError.
    css = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#03080f;--sur:#080f1a;--sur2:#0c1525;
  --b1:#0f2035;--b2:#1a3050;
  --tx:#8aa8c4;--txd:#2e4a65;--txh:#ddeeff;
  --cy:#1ab8d8;--gr:#0dcc6e;--re:#e8324a;--am:#e8a020;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}
html,body{min-height:100vh;background:var(--bg);color:var(--tx);
  font-family:var(--mono);font-size:14px;line-height:1.75}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,
  rgba(0,0,0,.04) 2px,rgba(0,0,0,.04) 4px)}
header{background:var(--sur);border-bottom:1px solid var(--b1);padding:13px 22px;
  position:sticky;top:0;z-index:50;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:8px}
.brand{font-family:var(--sans);font-weight:700;font-size:15px;color:var(--txh);
  display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--gr);
  box-shadow:0 0 8px var(--gr)}
.meta{font-size:10px;color:var(--txd)}
.back{font-size:11px;color:var(--cy);text-decoration:none;border:1px solid var(--b2);
  padding:4px 10px;border-radius:2px}
.back:hover{background:rgba(26,184,216,.08);border-color:var(--cy)}
.content{max-width:860px;margin:0 auto;padding:28px 22px}
h2{font-family:var(--sans);font-weight:700;font-size:17px;color:var(--txh);
  margin:36px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--b2)}
h2::before{content:'\\25C8  ';color:var(--am);font-size:13px}
h3{font-family:var(--sans);font-weight:600;font-size:11px;color:var(--cy);
  margin:18px 0 8px;letter-spacing:.5px;text-transform:uppercase}
p{margin:0 0 10px;line-height:1.75}
ul,ol{margin:0 0 10px 20px}
li{margin-bottom:4px;line-height:1.65}
strong{color:var(--txh);font-weight:600}
em{color:var(--am);font-style:normal}
table{width:100%;border-collapse:collapse;margin:8px 0 14px;font-size:12px}
th{background:var(--sur2);color:var(--txd);font-weight:400;letter-spacing:1px;
  text-transform:uppercase;font-size:10px;padding:7px 10px;text-align:left;
  border-bottom:1px solid var(--b2)}
td{padding:6px 10px;border-bottom:1px solid var(--b1);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td:first-child{color:var(--txd)}
code{background:var(--sur2);border:1px solid var(--b1);padding:1px 5px;
  border-radius:2px;font-family:var(--mono);font-size:12px;color:var(--cy)}
pre{background:var(--sur2);border:1px solid var(--b1);padding:14px;
  border-radius:3px;overflow-x:auto;margin:0 0 12px}
pre code{background:none;border:none;padding:0}
hr{border:none;border-top:1px solid var(--b1);margin:26px 0}
.sig-sb{color:#0dcc6e;font-weight:600}
.sig-b {color:#08a858;font-weight:600}
.sig-n {color:#e8a020;font-weight:600}
.sig-s {color:#d05068;font-weight:600}
.sig-ss{color:#e8324a;font-weight:600}
.pos{color:#0dcc6e}
.neg{color:#e8324a}
footer{border-top:1px solid var(--b1);padding:11px 22px;font-size:10px;
  color:var(--txd);display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}
@media(max-width:600px){.content{padding:14px 12px}h2{font-size:14px}}
"""

    page = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "<title>Market Intelligence Report</title>\n"
        "<link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&"
        "family=IBM+Plex+Sans:wght@300;600;700&display=swap\" rel=\"stylesheet\">\n"
        "<style>" + css + "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "  <div class=\"brand\"><div class=\"dot\"></div>Market Intelligence</div>\n"
        "  <div style=\"display:flex;align-items:center;gap:14px\">\n"
        "    <span class=\"meta\">Generated: " + timestamp + " UTC</span>\n"
        "    <a class=\"back\" href=\"index.html\">&#8592; Dashboard</a>\n"
        "  </div>\n"
        "</header>\n"
        "<div class=\"content\">\n"
        + body_html + "\n"
        "</div>\n"
        "<footer>\n"
        "  <span>Generated by Claude AI with live web search &middot; marketintell2000</span>\n"
        "  <span>Not financial advice.</span>\n"
        "</footer>\n"
        "</body>\n"
        "</html>\n"
    )
    return page


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Running marketintell2000...")

    try:
        report_md = run_research(client)
    except Exception as e:
        print("ERROR during API call: " + str(e), file=sys.stderr)
        sys.exit(1)

    if not report_md:
        print("ERROR: API returned empty response.", file=sys.stderr)
        sys.exit(1)

    print("  Converting to HTML...")
    body_html = md_to_html(report_md)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    page = build_page(body_html, timestamp)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    print("  Done. Saved to: " + output_path)


if __name__ == "__main__":
    main()
