"""
run_report.py  —  marketintell2000
Generates a daily market intelligence report and saves it as report.html.
"""

import os
import sys
import traceback
import anthropic
from datetime import datetime, timezone


# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS = [
    "S&P 500 (SPY)",
    "Nasdaq 100 (QQQ)",
    "Gold (XAU/USD)",
    "Bitcoin (BTC-USD)",
    "Crude Oil (WTI)",
    "US Treasuries (10-year yield)",
]


# ── PROMPTS ───────────────────────────────────────────────────────────────────
# The system prompt is large and static — perfect for prompt caching.
# Only the date in the user message changes daily.

SYSTEM_PROMPT = (
    "You are a professional market analyst. Your job is to produce a comprehensive "
    "daily market intelligence report in clean Markdown.\n\n"
    "When asked, search the web for current prices, news, and events for each asset.\n\n"
    "For EACH asset in the list, provide exactly this structure:\n\n"
    "## [Asset Name]\n\n"
    "### Prices & Changes\n"
    "Search for the latest price. Provide a Markdown table with four columns: "
    "Current Price | 1-Day Change (%) | 1-Week Change (%) | 1-Month Change (%)\n\n"
    "### Technical Signals\n"
    "Based on current price action and any available data:\n"
    "- RSI (14): value + signal (Overbought / Neutral / Oversold)\n"
    "- MACD: Bullish crossover / Bearish crossover / Neutral\n"
    "- Moving Averages (50 / 200): Above both / Mixed / Below both\n"
    "- Overall signal: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL\n\n"
    "### Latest News (last 48 hours)\n"
    "Search for recent news. List 3-5 items with a brief note on price impact.\n\n"
    "### Upcoming Events (next 14 days)\n"
    "List relevant economic releases, central bank meetings, earnings, options expiry "
    "or other scheduled events that could move this asset. "
    "Write 'No major events scheduled.' if none apply.\n\n"
    "### Key Price Drivers\n"
    "2-3 bullet points on the macro or sector forces currently driving or likely "
    "to move this asset.\n\n"
    "---\n\n"
    "After covering all assets, add a **Market Summary** section: 3-5 sentences on "
    "the overall market environment, dominant themes, and key risks to watch.\n\n"
    "Rules:\n"
    "- Use Markdown throughout. Use tables where appropriate.\n"
    "- Be specific — search for actual current prices and numbers, do not fabricate.\n"
    "- Keep each asset section concise. Avoid filler text.\n"
    "- If a data point is unavailable, write N/A rather than guessing."
)

ASSETS = [
    "S&P 500 (SPY)",
    "Nasdaq 100 (QQQ)",
    "Gold (XAU/USD)",
    "Bitcoin (BTC-USD)",
    "Crude Oil (WTI)",
    "US Treasuries (10-year yield)",
]


def build_user_message():
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    assets_str = "\n".join("- " + a for a in ASSETS)
    return (
        "Today is " + today + ". "
        "Please generate the market intelligence report for these assets:\n\n"
        + assets_str
    )


# ── API CALL ──────────────────────────────────────────────────────────────────
def run_research(client):
    user_msg = build_user_message()
    print("  Sending request to Anthropic API...")
    print("  User message: " + user_msg[:80] + "...")

    # Attempt 1: Haiku + web search + prompt caching
    try:
        response = _call_with_search(client, user_msg)
        print("  Web search call succeeded.")
        return _extract_text(response)
    except anthropic.RateLimitError:
        print("  Rate limit hit on web search — waiting 65s for window to reset...")
        import time; time.sleep(65)
        try:
            response = _call_with_search(client, user_msg)
            print("  Web search retry succeeded.")
            return _extract_text(response)
        except Exception as e:
            print("  Web search retry failed: " + str(e))
    except Exception as e:
        print("  Web search attempt failed: " + str(e))

    print("  Falling back to standard call (no live web search)...")

    # Attempt 2: Haiku without web search + prompt caching
    for attempt in range(3):
        try:
            response = _call_standard(client, user_msg)
            print("  Standard call succeeded.")
            return _extract_text(response)
        except anthropic.RateLimitError:
            wait = 65 * (attempt + 1)
            print("  Rate limit hit — waiting " + str(wait) + "s (attempt " + str(attempt+1) + "/3)...")
            import time; time.sleep(wait)
        except Exception as e:
            print("  Standard call failed: " + str(e))
            traceback.print_exc()
            raise

    raise RuntimeError("All attempts failed due to rate limits. Try again in a few minutes.")


def _call_with_search(client, user_msg):
    """
    Haiku + web search tool + cached system prompt.
    Tries beta header first (required by some SDK versions), falls back without.
    """
    messages = [{"role": "user", "content": user_msg}]
    tool_def = [{"type": "web_search_20250305", "name": "web_search"}]
    try:
        return _agentic_loop(client, messages, tool_def, use_beta=True)
    except anthropic.RateLimitError:
        raise  # propagate rate limits upward, don't retry here
    except Exception as e:
        print("    (beta attempt failed: " + str(e) + ", trying without beta)")
        return _agentic_loop(client, messages, tool_def, use_beta=False)


def _agentic_loop(client, messages, tools, use_beta=False):
    """Run the model with tool_use loop, up to 10 turns."""
    # Cached system prompt: the large static instructions are marked for caching.
    # After the first run, Anthropic caches these tokens — subsequent runs
    # are charged at cache read price (~10% of normal input price).
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    for turn in range(10):
        kwargs = dict(
            model="claude-haiku-4-5-20251001",
            max_tokens=4000,
            system=system,
            tools=tools,
            messages=messages,
        )

        if use_beta:
            response = client.beta.messages.create(
                betas=["web-search-2025-03-05", "prompt-caching-2024-07-31"],
                **kwargs
            )
        else:
            response = client.messages.create(**kwargs)

        print("  turn=" + str(turn)
              + " stop_reason=" + str(response.stop_reason)
              + " blocks=" + str(len(response.content)))

        # Log cache stats if available
        if hasattr(response, "usage"):
            u = response.usage
            cache_read  = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_write:
                print("  cache: read=" + str(cache_read)
                      + " write=" + str(cache_write)
                      + " (saved ~$" + "{:.4f}".format(cache_read * 0.000000025) + ")")

        if response.stop_reason == "end_turn":
            return response

        if response.stop_reason == "tool_use":
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
            continue

        return response

    return response


def _call_standard(client, user_msg):
    """Haiku without web search tools — uses training knowledge as fallback."""
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )


def _extract_text(response):
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
    in_ul = in_ol = in_table = in_pre = False
    thead_done = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:  out.append("</ul>");  in_ul = False
        if in_ol:  out.append("</ol>");  in_ol = False

    def close_table():
        nonlocal in_table, thead_done
        if in_table:
            out.append("</tbody></table>")
            in_table = thead_done = False

    def inline(t):
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"__(.+?)__",     r"<strong>\1</strong>", t)
        t = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", t)
        t = re.sub(r"`(.+?)`",       r"<code>\1</code>", t)
        for sig, cls in [
            ("STRONG BUY","sig-sb"),("STRONG SELL","sig-ss"),
            ("BUY","sig-b"),("SELL","sig-s"),("NEUTRAL","sig-n"),
        ]:
            t = t.replace(sig, '<span class="' + cls + '">' + sig + "</span>")
        t = re.sub(r"(\+[\d.]+\s*%)", r'<span class="pos">\1</span>', t)
        t = re.sub(r"(-[\d.]+\s*%)",  r'<span class="neg">\1</span>', t)
        return t

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            close_lists(); close_table()
            if not in_pre: out.append("<pre><code>"); in_pre = True
            else:           out.append("</code></pre>"); in_pre = False
            continue
        if in_pre: out.append(line); continue

        # HR
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line.strip()):
            close_lists(); close_table(); out.append("<hr>"); continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                close_lists(); out.append("<table>")
                in_table = True; thead_done = False
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                out.append("<tbody>"); thead_done = True
            elif not thead_done:
                out.append("<thead><tr>"
                    + "".join("<th>" + inline(c) + "</th>" for c in cells)
                    + "</tr></thead>")
            else:
                out.append("<tr>"
                    + "".join("<td>" + inline(c) + "</td>" for c in cells)
                    + "</tr>")
            continue
        else:
            close_table()

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            close_lists()
            tag = "h2" if len(m.group(1)) <= 2 else "h3"
            out.append("<" + tag + ">" + inline(m.group(2)) + "</" + tag + ">")
            continue

        # Unordered list
        m = re.match(r"^[-*+]\s+(.*)", line)
        if m:
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append("<li>" + inline(m.group(1)) + "</li>"); continue

        # Ordered list
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append("<li>" + inline(m.group(1)) + "</li>"); continue

        # Blank
        if not line.strip(): close_lists(); continue

        # Paragraph
        close_lists()
        out.append("<p>" + inline(line) + "</p>")

    close_lists(); close_table()
    if in_pre: out.append("</code></pre>")
    return "\n".join(out)


# ── HTML PAGE ─────────────────────────────────────────────────────────────────
def build_page(body_html, timestamp):
    css = (
        "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n"
        ":root{\n"
        "  --bg:#03080f;--sur:#080f1a;--sur2:#0c1525;\n"
        "  --b1:#0f2035;--b2:#1a3050;\n"
        "  --tx:#8aa8c4;--txd:#2e4a65;--txh:#ddeeff;\n"
        "  --cy:#1ab8d8;--gr:#0dcc6e;--re:#e8324a;--am:#e8a020;\n"
        "  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;\n"
        "}\n"
        "html,body{min-height:100vh;background:var(--bg);color:var(--tx);\n"
        "  font-family:var(--mono);font-size:14px;line-height:1.75}\n"
        "body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;\n"
        "  background:repeating-linear-gradient(0deg,transparent,transparent 2px,\n"
        "  rgba(0,0,0,.04) 2px,rgba(0,0,0,.04) 4px)}\n"
        "header{background:var(--sur);border-bottom:1px solid var(--b1);padding:13px 22px;\n"
        "  position:sticky;top:0;z-index:50;display:flex;align-items:center;\n"
        "  justify-content:space-between;flex-wrap:wrap;gap:8px}\n"
        ".brand{font-family:var(--sans);font-weight:700;font-size:15px;color:var(--txh);\n"
        "  display:flex;align-items:center;gap:8px}\n"
        ".dot{width:8px;height:8px;border-radius:50%;background:var(--gr);\n"
        "  box-shadow:0 0 8px var(--gr)}\n"
        ".meta{font-size:10px;color:var(--txd)}\n"
        ".back{font-size:11px;color:var(--cy);text-decoration:none;\n"
        "  border:1px solid var(--b2);padding:4px 10px;border-radius:2px}\n"
        ".back:hover{background:rgba(26,184,216,.08);border-color:var(--cy)}\n"
        ".content{max-width:860px;margin:0 auto;padding:28px 22px}\n"
        "h2{font-family:var(--sans);font-weight:700;font-size:17px;color:var(--txh);\n"
        "  margin:36px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--b2)}\n"
        "h3{font-family:var(--sans);font-weight:600;font-size:11px;color:var(--cy);\n"
        "  margin:18px 0 8px;letter-spacing:.5px;text-transform:uppercase}\n"
        "p{margin:0 0 10px;line-height:1.75}\n"
        "ul,ol{margin:0 0 10px 20px}\n"
        "li{margin-bottom:4px;line-height:1.65}\n"
        "strong{color:var(--txh);font-weight:600}\n"
        "em{color:var(--am);font-style:normal}\n"
        "table{width:100%;border-collapse:collapse;margin:8px 0 14px;font-size:12px}\n"
        "th{background:var(--sur2);color:var(--txd);font-weight:400;letter-spacing:1px;\n"
        "  text-transform:uppercase;font-size:10px;padding:7px 10px;text-align:left;\n"
        "  border-bottom:1px solid var(--b2)}\n"
        "td{padding:6px 10px;border-bottom:1px solid var(--b1);\n"
        "  font-variant-numeric:tabular-nums}\n"
        "tr:last-child td{border-bottom:none}\n"
        "td:first-child{color:var(--txd)}\n"
        "code{background:var(--sur2);border:1px solid var(--b1);padding:1px 5px;\n"
        "  border-radius:2px;font-family:var(--mono);font-size:12px;color:var(--cy)}\n"
        "pre{background:var(--sur2);border:1px solid var(--b1);padding:14px;\n"
        "  border-radius:3px;overflow-x:auto;margin:0 0 12px}\n"
        "pre code{background:none;border:none;padding:0}\n"
        "hr{border:none;border-top:1px solid var(--b1);margin:26px 0}\n"
        ".sig-sb{color:#0dcc6e;font-weight:600}\n"
        ".sig-b {color:#08a858;font-weight:600}\n"
        ".sig-n {color:#e8a020;font-weight:600}\n"
        ".sig-s {color:#d05068;font-weight:600}\n"
        ".sig-ss{color:#e8324a;font-weight:600}\n"
        ".pos{color:#0dcc6e}\n"
        ".neg{color:#e8324a}\n"
        "footer{border-top:1px solid var(--b1);padding:11px 22px;font-size:10px;\n"
        "  color:var(--txd);display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}\n"
        "@media(max-width:600px){.content{padding:14px 12px}h2{font-size:14px}}\n"
    )

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "<title>Market Intelligence Report \xe2\x80\x94 " + timestamp + "</title>\n"
        "<link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500"
        "&family=IBM+Plex+Sans:wght@300;600;700&display=swap\" rel=\"stylesheet\">\n"
        "<style>\n" + css + "</style>\n"
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
        + body_html +
        "\n</div>\n"
        "<footer>\n"
        "  <span>Generated by Claude AI with live web search &middot; marketintell2000</span>\n"
        "  <span>Not financial advice.</span>\n"
        "</footer>\n"
        "</body>\n"
        "</html>\n"
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=== marketintell2000 starting ===")
    print("Python version: " + sys.version)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    print("API key found (length=" + str(len(api_key)) + ")")

    # Check anthropic version
    try:
        print("anthropic version: " + anthropic.__version__)
    except Exception:
        pass

    client = anthropic.Anthropic(api_key=api_key)

    # Run research
    try:
        report_md = run_research(client)
    except Exception as e:
        print("FATAL ERROR during research:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if not report_md:
        print("ERROR: API returned empty text.", file=sys.stderr)
        sys.exit(1)

    print("  Report length: " + str(len(report_md)) + " chars")

    # Build HTML
    body_html = md_to_html(report_md)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    page = build_page(body_html, timestamp)

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)

    print("=== Done. Saved: " + out + " ===")


if __name__ == "__main__":
    main()
