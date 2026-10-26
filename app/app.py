"""
Chordify Web UI — Gradio-based interactive demo.

v5 changes:
  - Polished dark music-app aesthetic (deep navy / accent purple)
  - Compact badge-style metadata row instead of raw markdown table
  - Removed verbose pipeline description card
  - Cleaner tab labels and layout
  - All features, outputs, and backend logic unchanged
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import gradio as gr
except ImportError:
    print("[Chordify] Install gradio:  pip install gradio")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from chordify import Chordify

# ── Colour palette ────────────────────────────────────────────────────────
SECTION_COLORS = {
    "intro": "#7C3AED", "verse":  "#2563EB", "chorus": "#059669",
    "bridge":"#DC2626", "solo":   "#D97706", "outro":  "#6B7280",
    "break": "#DB2777", "inst":   "#0891B2", "start":  "#4B5563",
    "end":   "#374151",
}

# ── CSS ───────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── App shell ── */
.app-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 16px 48px;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 40px 0 28px;
    border-bottom: 1px solid #1e2433;
    margin-bottom: 32px;
}
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Panels ── */
.panel {
    background: #131720 !important;
    border: 1px solid #1e2433 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ── Upload area ── */
.upload-panel {
    background: #131720 !important;
    border: 1px solid #1e2433 !important;
    border-radius: 14px !important;
    padding: 24px !important;
}

/* ── Analyse button ── */
button.analyse-btn, button#analyse-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 0 !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
button.analyse-btn:hover { opacity: 0.88 !important; }

/* ── Metadata badges row ── */
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 18px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2033;
    border: 1px solid #2a3349;
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #cbd5e1;
    font-weight: 500;
}
.badge .label { color: #64748b; margin-right: 2px; }
.badge .val   { color: #e2e8f0; font-weight: 700; }
.badge.green  { border-color: #065f46; background: #052e16; color: #6ee7b7; }
.badge.amber  { border-color: #78350f; background: #1c0a00; color: #fcd34d; }

/* ── Tabs ── */
.tab-nav {
    border-bottom: 1px solid #1e2433 !important;
    background: transparent !important;
}
.tab-nav button {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
    background: transparent !important;
}

/* ── Markdown tables ── */
.prose table { width: 100%; border-collapse: collapse; }
.prose table th {
    background: #1a2033;
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 8px 12px;
    text-align: left;
}
.prose table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1e2433;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.prose table tr:last-child td { border-bottom: none; }

/* ── Textbox (ASCII tab) ── */
textarea, .textbox textarea {
    background: #0d0f14 !important;
    border: 1px solid #1e2433 !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-family: 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
}

/* ── Plot ── */
.plot-container { border-radius: 12px; overflow: hidden; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a3349; border-radius: 3px; }

/* ── Checkbox ── */
input[type=checkbox]:checked { accent-color: #7c3aed; }

/* ── Audio input ── */
.audio-container {
    background: #131720 !important;
    border: 1px dashed #2a3349 !important;
    border-radius: 12px !important;
}
"""

# ── Singleton pipeline ────────────────────────────────────────────────────
_pipeline: Chordify | None = None

def _get_pipeline() -> Chordify:
    global _pipeline
    if _pipeline is None:
        _pipeline = Chordify()
    return _pipeline


# ── Core analysis ─────────────────────────────────────────────────────────
def analyze_audio(audio_file, save_stems):
    if audio_file is None:
        empty = ("*Upload a track and click **Analyse**.*",
                 "—", "—", "—", "—", "", None, "", "", "")
        return empty

    try:
        result = _get_pipeline().analyze(audio_file, save_stems=save_stems)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        err = f"**Pipeline error:**\n```\n{tb}\n```"
        return (err, "—", "—", "—", "—", "", None, "", "", "")

    meta      = result["metadata"]
    beats     = result["beats"]
    chords    = result["chords"]
    sections  = result["sections"]
    lyrics    = result["lyrics"]
    all_words = result.get("words", [])
    tabs      = result.get("tabs", {})
    sync      = tabs.get("sync_view", [])

    # ── Method badges ─────────────────────────────────────────────────────
    cm  = meta.get("chord_method", "neural")
    bm  = meta.get("beat_method",  "neural")
    lm  = meta.get("lyrics_model", "Whisper")
    cb  = "🟢 Neural" if cm == "neural" else "🟡 DSP Chroma"
    bb  = "🟢 Neural" if bm == "neural" else "🟡 DSP Librosa"

    last_word_t  = max((w["end"] for w in all_words), default=0.0) if all_words else 0.0
    coverage_pct = (last_word_t / meta["duration"] * 100) if meta["duration"] > 0 else 0

    # Compact badge-style metadata (HTML)
    meta_md = _build_meta_badges(meta, cb, bb, lm, chords, lyrics, all_words, coverage_pct, last_word_t)

    # ── Chords ────────────────────────────────────────────────────────────
    if chords:
        chord_lines = ["| Start | End | Chord |", "|---|---|---|"]
        for c in chords[:60]:
            chord_lines.append(
                f"| `{c['start']:.1f}s` | `{c['end']:.1f}s` | **{c['label']}** |"
            )
        if len(chords) > 60:
            chord_lines.append(f"| … | … | *{len(chords)-60} more* |")
        chords_md = "\n".join(chord_lines)
    else:
        chords_md = "*No chords detected.*"

    # ── Beats ─────────────────────────────────────────────────────────────
    if beats:
        beat_lines = [f"`{b['time']:.2f}s`  {b['type']}" for b in beats[:80]]
        if len(beats) > 80:
            beat_lines.append(f"*… and {len(beats)-80} more*")
        beats_md = "\n".join(beat_lines)
    else:
        beats_md = "*No beats detected.*"

    # ── Sections ──────────────────────────────────────────────────────────
    if sections:
        sec_lines = ["| Start | End | Section |", "|---|---|---|"]
        for s in sections[:30]:
            sec_lines.append(
                f"| `{s['start']:.1f}s` | `{s['end']:.1f}s` | **{s['label'].upper()}** |"
            )
        secs_md = "\n".join(sec_lines)
    else:
        secs_md = "*No sections detected.*"

    # ── Lyrics ────────────────────────────────────────────────────────────
    if lyrics:
        lyr_lines = ["| Time | Text |", "|---|---|"]
        for l in lyrics:
            lyr_lines.append(f"| `{l['start']:.1f}s` | {l['text']} |")
        lyrics_md = "\n".join(lyr_lines)
    else:
        lyrics_md = (
            "*No lyrics detected.*\n\n"
            "If the track has vocals they should appear here — "
            "make sure the audio is not purely instrumental."
        )

    # ── Synchronized view ─────────────────────────────────────────────────
    sync_md  = _build_sync_markdown(sync, chords, all_words, meta["bpm"])

    # ── Word timestamps ───────────────────────────────────────────────────
    words_md = _build_words_markdown(all_words)

    # ── Diagnostics ───────────────────────────────────────────────────────
    diag_md  = _build_diagnostics_markdown(meta, chords, lyrics, all_words, beats)

    # ── ASCII tab ─────────────────────────────────────────────────────────
    ascii_tab = tabs.get("ascii_tab", "")

    # ── Timeline plot ─────────────────────────────────────────────────────
    fig = _make_timeline(chords, sections, meta["duration"])

    return (meta_md, beats_md, chords_md, secs_md, lyrics_md,
            ascii_tab, fig, sync_md, words_md, diag_md)


# ── Metadata HTML badges ──────────────────────────────────────────────────
def _build_meta_badges(meta, cb, bb, lm, chords, lyrics, words,
                       coverage_pct, last_word_t) -> str:
    dur  = meta.get("duration", 0)
    bpm  = meta.get("bpm", 0)
    key  = meta.get("key", "?")
    ks   = meta.get("key_score", 0)

    def badge(icon, label, val, cls=""):
        return (f'<span class="badge {cls}">'
                f'<span class="label">{icon} {label}</span>'
                f'<span class="val">{val}</span>'
                f'</span>')

    chord_cls = "green" if "Neural" in cb else "amber"
    beat_cls  = "green" if "Neural" in bb else "amber"

    row1 = (
        badge("⏱", "Duration", f"{dur:.0f}s") +
        badge("♩", "BPM", f"{bpm:.0f}") +
        badge("🎵", "Key", f"{key}  <span style='color:#64748b;font-weight:400;font-size:.75em'>score {ks}</span>") +
        badge("🎸", "Chords", str(len(chords))) +
        badge("🎤", "Lyrics", f"{len(lyrics)} phrases · {len(words)} words") +
        badge("📡", "Coverage", f"{coverage_pct:.0f}%")
    )
    row2 = (
        badge("🎼", "Chord engine", cb, chord_cls) +
        badge("🥁", "Beat engine",  bb, beat_cls) +
        badge("💬", "ASR model",    lm)
    )

    return (
        f'<div class="meta-row">{row1}</div>'
        f'<div class="meta-row">{row2}</div>'
    )


# ── Synchronized view ─────────────────────────────────────────────────────
def _build_sync_markdown(sync, chords, words, bpm) -> str:
    if not chords:
        return "*No chord data to synchronise.*"

    lines = ["### Synchronized Chord & Lyric Chart\n"]
    lines.append("| Time | Chord | Lyrics |")
    lines.append("|---|---|---|")

    for chord in chords:
        start = chord["start"]
        end   = chord["end"]
        label = chord["label"].replace(":maj", "").replace(":min", "m")
        t_str = f"{int(start//60)}:{int(start%60):02d} – {int(end//60)}:{int(end%60):02d}"

        matched = []
        for w in words:
            mid = (w["start"] + w["end"]) / 2.0
            if start <= mid < end:
                matched.append(w["word"])

        lyric_str = " ".join(matched).strip() or "—"
        lines.append(f"| `{t_str}` | **{label}** | {lyric_str} |")

    return "\n".join(lines)


# ── Word timestamps ───────────────────────────────────────────────────────
def _build_words_markdown(words) -> str:
    if not words:
        return "*No word-level timestamps available.*"

    lines = [f"**{len(words)} words detected**"]
    if words:
        lines.append(
            f"First: *{words[0]['word']}* @ {words[0]['start']:.2f}s  ·  "
            f"Last: *{words[-1]['word']}* @ {words[-1]['end']:.2f}s\n"
        )
    lines.append("| # | Word | Start | End | Confidence |")
    lines.append("|---|---|---|---|---|")
    for i, w in enumerate(words, 1):
        prob = f"{w.get('probability', 0):.0%}" if "probability" in w else "—"
        lines.append(f"| {i} | {w['word']} | `{w['start']:.2f}s` | `{w['end']:.2f}s` | {prob} |")

    return "\n".join(lines)


# ── Diagnostics ───────────────────────────────────────────────────────────
def _build_diagnostics_markdown(meta, chords, lyrics, words, beats) -> str:
    duration     = meta.get("duration", 0)
    last_word_t  = max((w["end"] for w in words), default=0.0) if words else 0.0
    first_word_t = min((w["start"] for w in words), default=0.0) if words else 0.0
    coverage     = (last_word_t / duration * 100) if duration > 0 else 0
    last_chord_t = max((c["end"] for c in chords), default=0.0) if chords else 0.0
    chord_cov    = (last_chord_t / duration * 100) if duration > 0 else 0

    lines = ["### Pipeline Diagnostics\n"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Audio duration | {duration:.1f}s |")
    lines.append(f"| ASR model | {meta.get('lyrics_model', '?')} |")
    lines.append(f"| Chord method | {meta.get('chord_method', '?')} |")
    lines.append(f"| Beat method | {meta.get('beat_method', '?')} |")
    lines.append(f"| Lyric phrases | {len(lyrics)} |")
    lines.append(f"| Lyric words | {len(words)} |")
    lines.append(f"| First word timestamp | {first_word_t:.2f}s |")
    lines.append(f"| Last word timestamp | {last_word_t:.2f}s |")
    lines.append(f"| Lyric coverage | {coverage:.0f}% |")
    lines.append(f"| Chord segments | {len(chords)} |")
    lines.append(f"| Chord coverage | {chord_cov:.0f}% |")
    lines.append(f"| Beat events | {len(beats)} |")
    lines.append(f"| BPM | {meta.get('bpm', '?')} |")
    lines.append(f"| Key | {meta.get('key', '?')} |")

    if coverage < 80 and len(words) > 0:
        lines.append(
            f"\n> ⚠️ Lyric coverage is {coverage:.0f}% — "
            "some vocal sections may not have been transcribed."
        )
    if len(words) == 0:
        lines.append(
            "\n> ⚠️ No words recognised. "
            "Check audio quality and Whisper model availability."
        )

    return "\n".join(lines)


# ── Timeline plot ─────────────────────────────────────────────────────────
def _make_timeline(chords, sections, duration):
    fig, (ax_sec, ax_chord) = plt.subplots(
        2, 1, figsize=(14, 3.6),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    BG      = "#0d0f14"
    PANEL   = "#131720"
    BORDER  = "#1e2433"

    fig.patch.set_facecolor(BG)

    # Section row
    ax_sec.set_facecolor(PANEL)
    for sec in sections:
        color = SECTION_COLORS.get(sec["label"], "#4B5563")
        w = sec["end"] - sec["start"]
        ax_sec.barh(0, w, left=sec["start"], height=0.72, color=color, alpha=0.9)
        if w > 3:
            ax_sec.text(sec["start"] + w / 2, 0, sec["label"],
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="600")
    ax_sec.set_xlim(0, max(duration, 1))
    ax_sec.set_yticks([])
    ax_sec.set_ylabel("Structure", color="#64748b", fontsize=8)
    ax_sec.tick_params(colors="#64748b", labelsize=7)
    for sp in ax_sec.spines.values():
        sp.set_edgecolor(BORDER)

    # Chord row
    ax_chord.set_facecolor(PANEL)
    cmap   = plt.cm.cool
    unique = list(dict.fromkeys(c["label"] for c in chords))
    clr    = {l: cmap(i / max(len(unique), 1)) for i, l in enumerate(unique)}

    for chord in chords:
        color = clr.get(chord["label"], (0.5, 0.5, 0.5, 0.8))
        w = chord["end"] - chord["start"]
        ax_chord.barh(0, w, left=chord["start"], height=0.65, color=color, alpha=0.92)
        if w > 3:
            short = chord["label"].replace(":maj", "").replace(":min", "m")
            ax_chord.text(chord["start"] + w / 2, 0, short,
                          ha="center", va="center", fontsize=7.5, color="white",
                          fontweight="600")

    ax_chord.set_xlim(0, max(duration, 1))
    ax_chord.set_yticks([])
    ax_chord.set_xlabel("Time (seconds)", color="#64748b", fontsize=8)
    ax_chord.set_ylabel("Chords", color="#64748b", fontsize=8)
    ax_chord.tick_params(colors="#64748b", labelsize=7)
    for sp in ax_chord.spines.values():
        sp.set_edgecolor(BORDER)

    patches = [
        mpatches.Patch(color=clr[l],
                       label=l.replace(":maj", "").replace(":min", "m"))
        for l in unique[:14]
    ]
    if patches:
        legend = ax_chord.legend(
            handles=patches, loc="upper right",
            fontsize=6.5, facecolor=PANEL, labelcolor="#cbd5e1",
            framealpha=0.9, ncol=7, edgecolor=BORDER,
        )

    plt.tight_layout(pad=0.8)
    return fig


# ── Gradio UI ─────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Chordify") as demo:

        # Header
        gr.HTML("""
        <div class="app-header">
          <h1>🎸 Chordify</h1>
          <p>AI-powered music analysis · chord recognition · lyrics · guitar tabs</p>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left column ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                audio_in = gr.Audio(
                    label="Drop a track here",
                    type="filepath",
                    elem_classes=["audio-container"],
                )
                save_cb = gr.Checkbox(
                    label="Save separated stems",
                    value=False,
                )
                run_btn = gr.Button(
                    "Analyse",
                    variant="primary",
                    size="lg",
                    elem_classes=["analyse-btn"],
                )

            # ── Right column ─────────────────────────────────────────────
            with gr.Column(scale=3):
                meta_out = gr.HTML("<p style='color:#4b5563;padding:12px 0'>Upload a track to get started.</p>")
                plot_out = gr.Plot(
                    label="",
                    elem_classes=["plot-container"],
                )

        # ── Tabs ─────────────────────────────────────────────────────────
        with gr.Tabs(elem_classes=["tab-nav"]):
            with gr.Tab("🎸 Guitar Tab"):
                tab_out = gr.Textbox(
                    label="",
                    lines=22,
                    max_lines=64,
                )

            with gr.Tab("🎵 Sync View"):
                sync_out = gr.Markdown()

            with gr.Tab("🎼 Chords"):
                chord_out = gr.Markdown()

            with gr.Tab("🥁 Beats"):
                beat_out = gr.Markdown()

            with gr.Tab("📐 Sections"):
                sec_out = gr.Markdown()

            with gr.Tab("🎤 Lyrics"):
                lyrics_out = gr.Markdown()

            with gr.Tab("📝 Words"):
                words_out = gr.Markdown()

            with gr.Tab("🔍 Diagnostics"):
                diag_out = gr.Markdown()

        # ── Footer ───────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;font-size:.72em;color:#2a3349;padding:24px 0 0">
          Chordify · AI Multimodal Music Analysis · Academic Project
        </div>
        """)

        run_btn.click(
            fn=analyze_audio,
            inputs=[audio_in, save_cb],
            outputs=[
                meta_out, beat_out, chord_out, sec_out,
                lyrics_out, tab_out, plot_out,
                sync_out, words_out, diag_out,
            ],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Chordify Web UI")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = build_ui()
    for port in range(args.port, args.port + 10):
        try:
            print(f"\n[Chordify] UI ready → http://localhost:{port}\n")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=args.share,
                show_error=True,
                theme=gr.themes.Base(
                    primary_hue="violet",
                    neutral_hue="slate",
                    font=gr.themes.GoogleFont("Inter"),
                ),
                css=CUSTOM_CSS,
            )
            break
        except OSError as e:
            if "10048" in str(e) or "Cannot find" in str(e):
                print(f"[Chordify] Port {port} busy, trying {port+1} …")
            else:
                raise


if __name__ == "__main__":
    main()
