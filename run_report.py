"""
run_report.py
Runs the marketintell2000 routine via Anthropic API with web search,
then saves the result as report.html in the repo root.
"""

import anthropic
import os
from datetime import datetime, timezone

# ─── CONFIG ───────────────────────────────────────────────────
TICKERS = [
    "S&P 500 (SPY)",
    "Nasdaq 100 (QQQ)",
    "Gold (GLD)",
    "Bitcoin (BTC-USD)",
    "Crude Oil (WTI)",
    "US Treasuries (10Y yield)",
]

PROMPT = f"""Today is {datetime.now(timezone.utc).strftime('%A, %B %d, %Y')}.

You are a professional market analyst. Search the web for current data and create a comprehensive
market intelligence report for the following assets:

{chr(10).join(f'- {t}' for t in TICKERS)}

For EACH asset, provide exactly the following structure:

## [Asset Name]

**Prices & Changes**
Search for the latest price and provide a table with:
- Current price
- 1-day change (% and direction)
- 1-week change (% and direction)
- 1-month change (% and direction)

**Technical Signals**
Based on current price action and any available technical data:
- RSI (14): value and signal (Overbought / Neutral / Oversold)
- MACD: signal (Bullish crossover / Bearish crossover / Neutral)
- Moving Averages: signal (Above MA50 & MA200 / Mixed / Below MA50 & MA200)
- Bollinger Bands: position (Upper band / Middle / Lower band)
- Overall signal: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL

**Latest News** (last 48 hours)
- List 3-5 most important news items with a brief description of price impact

**Upcoming Events** (next 14 days)
- List any relevant economic releases, earnings, Fed meetings, options expiry, or other
  scheduled events that could impact the price. If none, write "No major events scheduled."

**Key Price Drivers**
- 2-3 bullet points on macro or sector factors currently driving or likely to move this asset

---

After all assets, add a short **Market Summary** section (3-5 sentences) covering the overall
market environment, dominant themes, and key risks to watch.

Format your entire response in clean Markdown. Use tables where appropriate for prices.
Be specific with numbers — search for actual current prices and data.
"""

# ─── HTML TEMPLATE ────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Intelligence Report</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#03080f;--sur:#080f1a;--sur2:#0c1525;
  --b1:#0f2035;--b2:#1a3050;
  --tx:#8aa8c4;--txd:#2e4a65;--txh:#ddeeff;
  --cy:#1ab8d8;--gr:#0dcc6e;--re:#e8324a;--am:#e8a020;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}}
html,body{{min-height:100vh;background:var(--bg);color:var(--tx);font-family:var(--mono);font-size:14px;line-height:1.7}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.04) 2px,rgba(0,0,0,.04) 4px)}}
header{{background:var(--sur);border-bottom:1px solid var(--b1);padding:14px 24px;
  position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}}
.brand{{font-family:var(--sans);font-weight:700;font-size:15px;color:var(--txh);display:flex;align-items:center;gap:8px}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--gr);box-shadow:0 0 8px var(--gr)}}
.meta{{font-size:10px;color:var(--txd)}}
.back-link{{font-size:11px;color:var(--cy);text-decoration:none;border:1px solid var(--b2);
  padding:4px 10px;border-radius:2px;transition:all .15s}}
.back-link:hover{{background:rgba(26,184,216,.08);border-color:var(--cy)}}
.content{{max-width:900px;margin:0 auto;padding:28px 24px}}
h2{{font-family:var(--sans);font-weight:700;font-size:18px;color:var(--txh);
  margin:36px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--b2);
  display:flex;align-items:center;gap:8px}}
h2::before{{content:'◈';color:var(--am);font-size:14px}}
h3{{font-family:var(--sans);font-weight:600;font-size:13px;color:var(--cy);
  margin:18px 0 8px;letter-spacing:.5px;text-transform:uppercase;font-size:11px}}
p{{margin:0 0 12px;color:var(--tx);line-height:1.75}}
ul,ol{{margin:0 0 12px 20px;color:var(--tx)}}
li{{margin-bottom:5px;line-height:1.6}}
strong{{color:var(--txh);font-weight:600}}
em{{color:var(--am);font-style:normal}}
table{{width:100%;border-collapse:collapse;margin:10px 0 16px;font-size:12px}}
th{{background:var(--sur2);color:var(--txd);font-weight:400;letter-spacing:1px;
  text-transform:uppercase;font-size:10px;padding:7px 10px;text-align:left;
  border-bottom:1px solid var(--b2)}}
td{{padding:7px 10px;border-bottom:1px solid var(--b1);color:var(--tx);font-variant-numeric:tabular-nums}}
tr:last-child td{{border-bottom:none}}
td:first-child{{color:var(--txd)}}
code{{background:var(--sur2);border:1px solid var(--b1);padding:1px 5px;border-radius:2px;
  font-family:var(--mono);font-size:12px;color:var(--cy)}}
pre{{background:var(--sur2);border:1px solid var(--b1);padding:14px;border-radius:3px;
  overflow-x:auto;margin:0 0 14px}}
pre code{{background:none;border:none;padding:0}}
blockquote{{border-left:2px solid var(--cy);padding-left:14px;margin:0 0 12px;color:var(--txd)}}
hr{{border:none;border-top:1px solid var(--b1);margin:28px 0}}
.signal-buy{{color:#0dcc6e;font-weight:600}}
.signal-sell{{color:#e8324a;font-weight:600}}
.signal-neutral{{color:#e8a020;font-weight:600}}
.pos{{color:#0dcc6e}}.neg{{color:#e8324a}}
footer{{border-top:1px solid var(--b1);padding:12px 24px;font-size:10px;color:var(--txd);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}}
@media(max-width:600px){{.content{{padding:16px 14px}}h2{{font-size:15px}}}}
</style>
</head>
<body>
<header>
  <div class="brand"><div class="dot"></div>Market Intelligence</div>
  <div style="display:flex;align-items:center;gap:14px">
    <span class="meta">Generated: {timestamp} UTC</span>
    <a class="back-link" href="index.html">&#8592; Dashboard</a>
  </div>
</header>
<div class="content">
{body}
</div>
<footer>
  <span>Generated by Claude AI with web search · marketintell2000</span>
  <span>Not financial advice.</span>
</footer>
</body>
</html>"""

# ─── MARKDOWN → HTML ──────────────────────────────────────────
def md_to_html(text):
    """
    Minimal Markdown-to-HTML converter covering the subset Claude uses.
    For production use you could swap this for `mistune` or `markdown` package.
    """
    import re
    lines = text.split('\n')
    html = []
    in_table = False
    in_ul = False
    in_ol = False
    in_pre = False
    table_header_done = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html.append('</ul>')
            in_ul = False
        if in_ol:
            html.append('</ol>')
            in_ol = False

    def inline(t):
        # Bold
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = re.sub(r'__(.+?)__',     r'<strong>\1</strong>', t)
        # Italic
        t = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', t)
        # Code
        t = re.sub(r'`(.+?)`',       r'<code>\1</code>', t)
        # Signals coloring
        for word, cls in [('STRONG BUY','signal-buy'),('BUY','signal-buy'),
                          ('STRONG SELL','signal-sell'),('SELL','signal-sell'),('NEUTRAL','signal-neutral')]:
            t = t.replace(word, f'<span class="{cls}">{word}</span>')
        # Positive/negative % 
        t = re.sub(r'(\+[\d.]+%)', r'<span class="pos">\1</span>', t)
        t = re.sub(r'(-[\d.]+%)',  r'<span class="neg">\1</span>', t)
        return t

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.startswith('```'):
            if not in_pre:
                close_lists()
                if in_table: html.append('</table>'); in_table=False
                html.append('<pre><code>')
                in_pre = True
            else:
                html.append('</code></pre>')
                in_pre = False
            i += 1
            continue
        if in_pre:
            html.append(line)
            i += 1
            continue

        # HR
        if re.match(r'^-{3,}$', line.strip()) or re.match(r'^\*{3,}$', line.strip()):
            close_lists()
            if in_table: html.append('</table>'); in_table=False
            html.append('<hr>')
            i += 1
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                close_lists()
                html.append('<table>')
                in_table = True
                table_header_done = False
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.match(r'^[-:]+$', c) for c in cells if c):
                html.append('<tbody>')
                table_header_done = True
            elif not table_header_done:
                html.append('<thead><tr>'+''.join(f'<th>{inline(c)}</th>' for c in cells)+'</tr></thead>')
            else:
                html.append('<tr>'+''.join(f'<td>{inline(c)}</td>' for c in cells)+'</tr>')
            i += 1
            continue
        elif in_table:
            html.append('</tbody></table>')
            in_table = False
            table_header_done = False

        # Headings
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            close_lists()
            level = len(m.group(1))
            tag = f'h{min(level+1, 4)}'  # shift: ## → h3, ### → h3
            if level == 2:
                html.append(f'<h2>{inline(m.group(2))}</h2>')
            else:
                html.append(f'<h3>{inline(m.group(2))}</h3>')
            i += 1
            continue

        # Unordered list
        m = re.match(r'^[-*+]\s+(.*)', line)
        if m:
            if in_ol: html.append('</ol>'); in_ol = False
            if not in_ul: html.append('<ul>'); in_ul = True
            html.append(f'<li>{inline(m.group(1))}</li>')
            i += 1
            continue

        # Ordered list
        m = re.match(r'^\d+\.\s+(.*)', line)
        if m:
            if in_ul: html.append('</ul>'); in_ul = False
            if not in_ol: html.append('<ol>'); in_ol = True
            html.append(f'<li>{inline(m.group(1))}</li>')
            i += 1
            continue

        # Blockquote
        if line.startswith('>'):
            close_lists()
            html.append(f'<blockquote>{inline(line[1:].strip())}</blockquote>')
            i += 1
            continue

        # Blank line
        if not line.strip():
            close_lists()
            i += 1
            continue

        # Paragraph
        close_lists()
        html.append(f'<p>{inline(line)}</p>')
        i += 1

    close_lists()
    if in_table:
        html.append('</tbody></table>')
    if in_pre:
        html.append('</code></pre>')

    return '\n'.join(html)


# ─── MAIN ─────────────────────────────────────────────────────
def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    print("Running market intelligence research...")

    messages = [{"role": "user", "content": PROMPT}]

    # Agentic loop — handles multiple web search tool calls
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

        print(f"  stop_reason: {response.stop_reason} | blocks: {len(response.content)}")

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        # If tool_use, feed back tool results and continue
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [],   # server-side tool — results already embedded
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — exit loop
            break

    # Extract final text from last assistant message
    report_md = ""
    for block in response.content:
        if hasattr(block, "text"):
            report_md += block.text

    if not report_md.strip():
        raise ValueError("No text output received from the API")

    print(f"  Report length: {len(report_md)} characters")

    # Convert markdown to HTML
    body_html = md_to_html(report_md)

    # Stamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    # Render full page
    html = HTML_TEMPLATE.format(
        body=body_html,
        timestamp=timestamp,
    )

    # Write output
    output_path = os.path.join(os.path.dirname(__file__), "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
