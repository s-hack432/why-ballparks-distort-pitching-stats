"""Step 9: figures (PNG) in outputs/figures/.

Style: light chart surface with explicit background, Montserrat text, hairline solid
gridlines, and thin marks. The three channels always use the first three validated categorical
slots: strikeout/walk = blue, contact quality = orange, conversion = aqua. Every pair stays
distinguishable under simulated color-vision deficiency. Aqua is below
3:1 contrast on the surface, so every chart that uses it carries a legend and direct labels, and
the README carries the table view. One-series charts use emphasis: the park the story is about in
the accent color, the rest in gray. Titles state the takeaway. Montserrat (SIL Open Font License)
is bundled in assets/fonts/montserrat/, so the figures look the same on any machine.

    fig1_park_channels.png      park channel breakdown (centerpiece)
    fig2_pitch_movement.png     movement change by park, vs altitude, and vs contact rate
    fig3_carry.png              carry by park, vs conversion, and home runs vs fence distance
    fig4_spray_charts.png       where balls beat their xwOBA, selected parks
    fig5_savant_validation.png  our estimates vs Savant's park factors
    fig6_case_studies.png       three pitchers, three channels
    fig7_actual_vs_neutral.png  actual vs park-neutral wOBA, all qualifying pitchers
    fig8_pooled_vs_2026.png     robustness: 2024-2026 vs 2026-only park totals
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from .common import get_logger  # noqa: E402
from .config import DATA_PROCESSED, OTHER_PARK, OUT_FIGURES, OUT_TABLES, ROOT  # noqa: E402

log = get_logger("plots", "step9_plots.log")

# ---- Tokens (reference palette, light mode) --------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPH = "#c3c2b7"       # de-emphasis gray for context marks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CH_COLORS = {"kbb": BLUE, "contact": ORANGE, "conv": AQUA}
CH_LABELS = {"kbb": "Strikeout/walk", "contact": "Contact quality", "conv": "Conversion"}
ROLE = {"SP": ("Starters (400+ BF)", BLUE), "RP": ("Relievers (150+ BF)", ORANGE)}
DPI = 150
TOP_IN = 1.0             # inches reserved above the plotting area for title + subtitle

# Montserrat ships with the project instead of being expected on the system. DejaVu Sans (bundled
# with matplotlib) stays second in the list as a per-glyph fallback.
FONT_DIR = ROOT / "assets" / "fonts" / "montserrat"
for _ttf in sorted(FONT_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))
if "Montserrat" not in {f.name for f in font_manager.fontManager.ttflist}:
    raise SystemExit(f"Montserrat not found: expected Montserrat-*.ttf files in {FONT_DIR}")

plt.rcParams.update({
    # Draw at the save resolution so text measured for layout (titles, wrapping, label placement)
    # has the same size as in the saved PNG.
    "figure.dpi": DPI,
    "font.family": "sans-serif", "font.sans-serif": ["Montserrat", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
})


# ---- Helpers ---------------------------------------------------------------------------------
def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def panel_title(ax, text):
    """Left-aligned panel title, wrapped word by word to the panel's measured width."""
    r = ax.figure.canvas.get_renderer()
    ax.apply_aspect()                   # equal-aspect panels shrink to their final box here
    width = ax.get_window_extent(r).width
    t = ax.set_title(text, loc="left", fontsize=10, color=INK2, pad=8)
    lines = [""]
    for word in text.split():
        t.set_text(f"{lines[-1]} {word}".lstrip())
        if lines[-1] and t.get_window_extent(r).width > width:
            lines.append(word)
        else:
            lines[-1] = t.get_text()
    t.set_text("\n".join(lines))


def top_frac(fig, inches=TOP_IN):
    return 1 - inches / fig.get_figheight()


def titles(fig, title, subtitle=None):
    """Title and subtitle at fixed inch offsets from the top; shrunk to fit the width, keeping the
    same 0.12-inch margin on the right as on the left."""
    h, w = fig.get_figheight(), fig.get_figwidth()
    t = fig.text(0.12 / w, 1 - 0.14 / h, title, ha="left", va="top", fontsize=14, weight="bold", color=INK)
    s = fig.text(0.12 / w, 1 - 0.48 / h, subtitle, ha="left", va="top", fontsize=10, color=INK2) if subtitle else None
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for txt, size in ((t, 14), (s, 10)):
        while txt is not None and txt.get_window_extent(r).x1 > fig.bbox.x1 - 0.12 * fig.dpi and size > 7:
            size -= 0.5
            txt.set_fontsize(size)


def place_labels(ax, xs, ys, names, avoid_x=None, avoid_y=None, fontsize=8, color=INK2):
    """Greedy label placement: try offsets around each point and keep the first one that stays
    inside the axes and overlaps neither an earlier label nor any plotted point; otherwise fall
    back to the first one that at least clears the earlier labels (in a dense cloud a label has
    to sit on some points), and failing that to the first offset."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r)
    pts = []
    if avoid_x is not None:
        P = ax.transData.transform(np.c_[np.asarray(avoid_x, float), np.asarray(avoid_y, float)])
        pts = [Bbox.from_bounds(px - 4, py - 4, 8, 8) for px, py in P]
    placed = []
    cands = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
             (-5, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
             (14, 10, "left", "bottom"), (14, -10, "left", "top"), (-14, 10, "right", "bottom"),
             (-14, -10, "right", "top"), (0, 18, "center", "bottom"), (0, -18, "center", "top"),
             (26, 18, "left", "bottom"), (-26, -18, "right", "top"), (26, -18, "left", "top"),
             (-26, 18, "right", "bottom")]
    for x, y, name in zip(xs, ys, names):
        px, py = ax.transData.transform((x, y))
        own = Bbox.from_bounds(px - 4, py - 4, 8, 8)
        ok = backup = None
        for dx, dy, ha, va in cands:
            far = abs(dx) > 6 or abs(dy) > 8
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=color, zorder=6,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3)
                            if far else None)
            bb = t.get_window_extent(r).expanded(1.04, 1.12)
            inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
            label_clash = any(bb.overlaps(p) for p in placed)
            point_clash = any(bb.overlaps(p) for p in pts if not np.allclose(p.bounds, own.bounds))
            if inside and not label_clash and not point_clash:
                ok = t
                break
            if inside and not label_clash and backup is None:
                backup = t      # keep it until a clean spot turns up
                continue
            t.remove()
        if ok is None and backup is not None:   # clear of other labels, on top of some points
            ok, backup = backup, None
        if backup is not None:
            backup.remove()
        if ok is None:   # last resort: first candidate even if it touches something
            dx, dy, ha, va = cands[0]
            ok = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                             fontsize=fontsize, color=color, zorder=6)
        placed.append(ok.get_window_extent(r))


def save(fig, name):
    fig.savefig(OUT_FIGURES / name, dpi=DPI)
    plt.close(fig)
    log.info("saved %s", name)


def label_of(row):
    team = row.get("home_team")
    return f"{row['venue_short']} ({team})" if isinstance(team, str) and team else row["venue_short"]


def dot_ci(ax, d, col, lo, hi, emphasis="Coors Field", hollow=None):
    """Dot-and-interval rows (one per park), emphasis on one park, the rest gray."""
    for i, (_, r) in enumerate(d.iterrows()):
        hi_ = r["venue_short"] == emphasis
        c = BLUE if hi_ else MUTED
        ax.plot([r[lo], r[hi]], [i, i], color=c, lw=1.2, zorder=2)
        filled = True if hollow is None else not r[hollow]
        ax.scatter(r[col], i, s=40 if hi_ else 24, color=c if filled else SURFACE, edgecolor=c,
                   linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_ylim(-0.7, len(d) - 0.3)


def scatter_emph(ax, x, y, names, emphasis="Coors Field"):
    is_e = np.asarray(names) == emphasis
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x[~is_e], y[~is_e], s=28, color=MUTED, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.scatter(x[is_e], y[is_e], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=4)


def trend(ax, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=AXIS, lw=1.2, zorder=2)
    return np.corrcoef(x[ok], y[ok])[0, 1]


# --------------------------------------------------------------------------------------------
# 1. Park channel breakdown
# --------------------------------------------------------------------------------------------
def fig_park_channels():
    d = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    d = d[(d["fit"] == "pooled") & (d["park_key"] != OTHER_PARK)].sort_values("total_pts")
    n = len(d)
    fig, ax = plt.subplots(figsize=(11, 0.33 * n + 1.9))
    fig.subplots_adjust(left=0.25, right=0.93, top=top_frac(fig, 1.05), bottom=0.7 / fig.get_figheight())
    style(ax, "x")
    h = 0.56
    for i, (_, r) in enumerate(d.iterrows()):
        pos = neg = 0.0
        for ch in ("kbb", "contact", "conv"):
            v = r[f"{ch}_pts"]
            ax.barh(i, v, left=pos if v >= 0 else neg, height=h, color=CH_COLORS[ch], edgecolor=SURFACE,
                    linewidth=1.0, zorder=2)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.plot([r["total_lo"], r["total_hi"]], [i, i], color=INK, lw=1.0, zorder=3)
        ax.scatter(r["total_pts"], i, s=26, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.text(max(r["total_hi"], pos) + 1.0, i, f"{r['total_pts']:+.0f}", va="center", ha="left",
                fontsize=8.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label_of(r) for _, r in d.iterrows()], fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Effect on wOBA allowed vs. a league-average park (wOBA points; 1 point = .001)")
    handles = [Patch(color=CH_COLORS[c], label=CH_LABELS[c]) for c in CH_COLORS]
    handles.append(Line2D([0], [0], color=INK, marker="o", lw=1.0, markersize=5, label="Total, 95% interval"))
    ax.legend(handles=handles, loc="lower right")
    top, bot = d.iloc[-1], d.iloc[0]
    titles(fig, f"{top['venue_short']} adds {top['total_pts']:.0f} points of wOBA, mostly through conversion; "
                f"{bot['venue_short']} takes away {abs(bot['total_pts']):.0f}",
           "Each park's effect on wOBA allowed, split into three channels (Shapley decomposition). 2024-2026 "
           "pooled; the same pitchers and batters compared across parks")
    save(fig, "fig1_park_channels.png")


# --------------------------------------------------------------------------------------------
# 2. Pitch movement
# --------------------------------------------------------------------------------------------
def fig_pitch_movement():
    m = pd.read_csv(OUT_TABLES / "mechanism_pitch_movement.csv")
    m = m[m["pitches"] >= 1000].sort_values("d_toward", ascending=False)
    attrs = pd.read_csv(OUT_TABLES / "mechanism_park_attributes.csv", dtype={"park_key": str}).set_index("park_key")
    fig = plt.figure(figsize=(15, 0.25 * len(m) + 1.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.3, left=0.12, right=0.985,
                          top=top_frac(fig, 1.1), bottom=0.95 / fig.get_figheight())
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    style(a1, "x")
    m = m.assign(lo=m["d_toward"] - 1.96 * m["d_toward_se"], hi=m["d_toward"] + 1.96 * m["d_toward_se"],
                 neutral=~m["modeled_park"])
    dot_ci(a1, m, "d_toward", "lo", "hi", hollow="neutral")
    a1.set_yticklabels([v + ("" if mp else " *") for v, mp in zip(m["venue_short"], m["modeled_park"])], fontsize=8.5)
    a1.axvline(0, color=AXIS, lw=1.0)
    a1.set_xlabel("Movement vs. the same pitcher's same pitch elsewhere (inches)")
    panel_title(a1, "Movement change along each pitch's usual direction")
    fig.text(0.12 / fig.get_figwidth(), 0.1 / fig.get_figheight(), "* One-off neutral site (hollow marker). "
             "Mexico City's Estadio Alfredo Harp Helú sits at about 7,300 ft.", fontsize=8, color=MUTED, va="bottom")

    p = attrs.dropna(subset=["elevation_ft", "d_toward"])
    style(a2, "both")
    r_el = trend(a2, p["elevation_ft"], p["d_toward"])
    scatter_emph(a2, p["elevation_ft"], p["d_toward"], p["venue_short"])
    lab = p[(p["venue_short"].isin(["Coors Field", "Chase Field", "Truist Park", "Tropicana Field", "Kauffman Stadium"]))]
    place_labels(a2, lab["elevation_ft"], lab["d_toward"], lab["venue_short"], p["elevation_ft"], p["d_toward"])
    nc = p["venue_short"] != "Coors Field"
    r_el_nc = np.corrcoef(p.loc[nc, "elevation_ft"], p.loc[nc, "d_toward"])[0, 1]
    a2.set_xlabel("Elevation (feet, MLB Stats API)")
    a2.set_ylabel("Movement change (inches)")
    panel_title(a2, f"Thinner air, flatter pitches: r = {r_el:.2f} (without Coors {r_el_nc:.2f})")

    q = attrs.dropna(subset=["contact_per_swing", "d_toward"])
    style(a3, "both")
    x, y = q["d_toward"], 100 * q["contact_per_swing"]
    r_all = trend(a3, x, y)
    scatter_emph(a3, x, y, q["venue_short"])
    nc = q["venue_short"] != "Coors Field"
    r_nc = np.corrcoef(x[nc], y[nc])[0, 1]
    pick = set(q.loc[y.abs().nlargest(3).index, "venue_short"]) | set(q.loc[x.abs().nlargest(3).index, "venue_short"])
    lab = q[q["venue_short"].isin(pick)]
    place_labels(a3, lab["d_toward"], 100 * lab["contact_per_swing"], lab["venue_short"], x, y)
    a3.set_xlabel("Movement change (inches)")
    a3.set_ylabel("Park effect on contact per swing (percentage points)")
    panel_title(a3, f"Flatter pitches, more contact: r = {r_all:.2f} (without Coors {r_nc:.2f})")
    cf = m[m["venue_short"] == "Coors Field"].iloc[0]
    titles(fig, f"Pitches lose {abs(cf['d_toward']):.1f} inches of movement at Coors Field, the thin-air effect, "
                "and batters miss less there",
           "Leave-one-park-out comparison within pitcher x season x pitch type, 2024-2026. Contact per swing from a "
           "pitcher- and batter-adjusted swing model; 95% intervals")
    save(fig, "fig2_pitch_movement.png")


# --------------------------------------------------------------------------------------------
# 3. Carry
# --------------------------------------------------------------------------------------------
def fig_carry():
    c = pd.read_csv(OUT_TABLES / "mechanism_carry.csv", dtype={"park_key": str})
    c = c[c["park_key"] != OTHER_PARK].sort_values("carry_adj")
    attrs = pd.read_csv(OUT_TABLES / "mechanism_park_attributes.csv", dtype={"park_key": str})
    fig = plt.figure(figsize=(15, 0.25 * len(c) + 1.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.3, left=0.12, right=0.985,
                          top=top_frac(fig, 1.1), bottom=0.75 / fig.get_figheight())
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    style(a1, "x")
    dot_ci(a1, c, "carry_adj", "carry_adj_lo", "carry_adj_hi")
    a1.set_yticklabels(c["venue_short"], fontsize=8.5)
    a1.axvline(0, color=AXIS, lw=1.0)
    a1.xaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 5, 10]))   # whole feet, not 2.5
    a1.set_xlabel("Extra projected distance, same EV, launch angle and direction (ft)")
    panel_title(a1, "Carry of fly balls and line drives, 95% interval")

    style(a2, "both")
    x, y = c["carry_adj"], c["conv_pts"]
    r = trend(a2, x, y)
    a2.axhline(0, color=AXIS, lw=0.8)
    scatter_emph(a2, x, y, c["venue_short"])
    b = np.polyfit(x, y, 1)
    resid = y - np.polyval(b, x)
    pick = set(c.loc[resid.abs().nlargest(6).index, "venue_short"]) | {"Coors Field", "Fenway Park"}
    lab = c[c["venue_short"].isin(pick)]
    place_labels(a2, lab["carry_adj"], lab["conv_pts"], lab["venue_short"], x, y)
    a2.set_xlabel("Carry (feet)")
    a2.set_ylabel("Conversion channel (wOBA points)")
    panel_title(a2, f"Carry explains {100 * r ** 2:.0f}% of the spread in conversion")

    p = attrs.dropna(subset=["carry_adj", "hr"])
    style(a3, "both")
    hx, hy = p["carry_adj"], 100 * p["hr"]
    rh = trend(a3, hx, hy)
    a3.axhline(0, color=AXIS, lw=0.8)
    scatter_emph(a3, hx, hy, p["venue_short"])
    pick = set(p.loc[hy.nlargest(3).index, "venue_short"]) | set(p.loc[hy.nsmallest(2).index, "venue_short"]) | \
        {"Coors Field", "Kauffman Stadium", "Chase Field"}
    lab = p[p["venue_short"].isin(pick)]
    place_labels(a3, lab["carry_adj"], 100 * lab["hr"], lab["venue_short"], hx, hy)
    f = p.dropna(subset=["fence_mean_ft"])
    rf = np.corrcoef(f["fence_mean_ft"], f["hr"])[0, 1]
    a3.set_xlabel("Carry (feet)")
    a3.set_ylabel("Park effect on home runs per ball in play (pct. points)")
    panel_title(a3, f"Home runs barely track carry (r = {rh:.2f}) or mean fence distance (r = {rf:.2f})")
    cf = c[c["venue_short"] == "Coors Field"].iloc[0]
    titles(fig, f"The ball carries {cf['carry_adj']:.0f} feet farther at Coors Field, but carry explains only a "
                "third of the conversion channel and almost none of the home-run differences",
           "Carry = residual of projected distance from a gradient-boosted model (exit velocity, launch angle, spray "
           "angle, batter side, season), pitcher- and batter-adjusted, 2024-2026")
    save(fig, "fig3_carry.png")


# --------------------------------------------------------------------------------------------
# 4. Spray charts
# --------------------------------------------------------------------------------------------
def fig_spray():
    sel = pd.read_parquet(DATA_PROCESSED / "spray_selected.parquet")
    zones = pd.read_csv(OUT_TABLES / "mechanism_overperformer_zones.csv")
    parks = pd.read_csv(OUT_TABLES / "spray_chart_parks.csv")["venue_short"].tolist()
    dec = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    dec = dec[dec["fit"] == "pooled"].set_index("venue_short")
    k = len(parks)
    fig = plt.figure(figsize=(3.3 * k, 6.6))
    gs = fig.add_gridspec(2, k, height_ratios=[1.55, 1], hspace=0.28, wspace=0.08,
                          left=0.055, right=0.995, top=top_frac(fig, 1.45), bottom=0.07)
    cats = {"single": ("Single", BLUE), "xbh": ("Double or triple", ORANGE), "home_run": ("Home run", AQUA)}
    first_bar = None
    zmax = 0
    for j, pk in enumerate(parks):
        z = zones[zones["venue_short"] == pk]
        zmax = max(zmax, (100 * z.groupby("direction")["overperformers"].sum()
                          / z.groupby("direction")["bip"].sum()).max())
    for j, pk in enumerate(parks):
        s = sel[sel["venue_short"] == pk]
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(s["x_ft"], s["y_ft"], s=1.0, color=GRID, alpha=0.6, linewidths=0, rasterized=True, zorder=1)
        o = s[s["over"]]
        typ = np.where(o["events"] == "home_run", "home_run",
                       np.where(o["events"].isin(["double", "triple"]), "xbh", "single"))
        for key, (lab, col) in cats.items():
            sub = o[typ == key]
            ax.scatter(sub["x_ft"], sub["y_ft"], s=4.0, color=col, linewidths=0, alpha=0.85, zorder=2,
                       rasterized=True)
        for sgn in (-1, 1):
            ax.plot([0, sgn * 330 / np.sqrt(2)], [0, 330 / np.sqrt(2)], color=AXIS, lw=0.8, zorder=0)
        ax.set_xlim(-330, 330)
        ax.set_ylim(-20, 470)
        ax.set_aspect("equal")
        ax.axis("off")
        rate = 100 * s["over"].mean()
        ax.set_title(f"{pk}\nconversion {dec.loc[pk, 'conv_pts']:+.0f} pts; {rate:.1f}% beat xwOBA",
                     fontsize=9.5, color=INK)
        z = zones[zones["venue_short"] == pk]
        order = ["LF", "LCF", "CF", "RCF", "RF"]
        g = z.assign(exp_n=z["bip"] * z["over_rate_league"]).groupby("direction")[["overperformers", "bip", "exp_n"]].sum()
        g = g.reindex(order)
        park_rate, league_rate = 100 * g["overperformers"] / g["bip"], 100 * g["exp_n"] / g["bip"]
        bx = fig.add_subplot(gs[1, j], sharey=first_bar)
        first_bar = first_bar or bx
        style(bx, "y")
        xi = np.arange(len(order))
        bx.bar(xi - 0.18, league_rate, width=0.34, color=DEEMPH, zorder=2, label="League, same zones")
        bx.bar(xi + 0.18, park_rate, width=0.34, color=BLUE, zorder=2, label="This park")
        bx.set_xticks(xi)
        bx.set_xticklabels(order, fontsize=8.5)
        bx.set_ylim(0, zmax * 1.12)
        if j == 0:
            bx.set_ylabel("% beating xwOBA by 0.5+")
        else:
            bx.tick_params(labelleft=False)
        if j == k - 1:
            bx.legend(loc="upper right", fontsize=8)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="none", markersize=6,
                      label=l) for l, c in cats.values()]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=GRID, markeredgecolor="none",
                          markersize=6, label="All other balls in play"))
    fig.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(0.995, top_frac(fig, 0.95)), fontsize=9)
    titles(fig, "Where balls in play beat their xwOBA: off Fenway's left-field wall, into Yankee Stadium's short "
                "right field, and all over Coors Field's huge outfield",
           "Balls in play worth at least 0.5 more wOBA than expected from exit velocity and launch angle, 2024-2026, "
           "plotted where fielded or landed (feet). Bottom: rate by direction, park vs. league in the same distance bands")
    save(fig, "fig4_spray_charts.png")


# --------------------------------------------------------------------------------------------
# 5. Savant validation
# --------------------------------------------------------------------------------------------
def fig_savant():
    v = pd.read_csv(OUT_TABLES / "savant_validation.csv")
    s = pd.read_csv(OUT_TABLES / "savant_validation_summary.csv")
    v = v[v["fit"] == "pooled"]
    s = s[s["fit"] == "pooled"].set_index("comparison")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.18, top=top_frac(fig, 1.15), bottom=0.09)
    specs = [("wobacon_idx", "index_wobacon", "wOBA on contact (contact + conversion)", "wOBA on contact"),
             ("hr_idx", "index_hr", "Home runs (ours: per ball in play)", "Home runs")]
    for ax, (ours, sv, key, lab) in zip(axes, specs):
        style(ax, "both")
        x, y = v[sv], v[ours]
        lo, hi = min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(x, y, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        diff = (y - x).abs()
        pick = set(v.loc[diff.nlargest(4).index, "venue_short"]) | {"Coors Field", "Fenway Park", "Yankee Stadium"}
        lab_rows = v[v["venue_short"].isin(pick)]
        place_labels(ax, lab_rows[sv], lab_rows[ours], lab_rows["venue_short"], x, y)
        ax.set_xlabel("Savant park factor, 2024-2026 (100 = average)")
        ax.set_ylabel("This project, same scale")
        st = s.loc[key]
        panel_title(ax, f"{lab}: r = {st['pearson']:.2f} (without Coors {st['pearson_without_coors']:.2f}), "
                        f"{int(st['n_parks'])} parks")
    titles(fig, "The channel estimates line up with Baseball Savant's published park factors",
           "Savant's factors were used only for this check. Diagonal = perfect agreement. Sutter Health Park, "
           "Steinbrenner Field and the Coliseum have no Savant 2024-2026 factor")
    save(fig, "fig5_savant_validation.png")


# --------------------------------------------------------------------------------------------
# 6. Case studies
# --------------------------------------------------------------------------------------------
def fig_case_studies():
    cs = pd.read_csv(OUT_TABLES / "pitcher_case_studies.csv")
    ex = pd.read_csv(OUT_TABLES / "pitcher_exposure_2026.csv").set_index("pitcher")
    dec = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    dec = dec[dec["fit"] == "pooled"]
    tot = dict(zip(dec["venue_short"], dec["total_pts"]))
    tot["Other"] = dec.loc[dec["park_key"] == OTHER_PARK, "total_pts"].iloc[0]
    fig = plt.figure(figsize=(13, 3.0 * len(cs) + 1.3))
    gs = fig.add_gridspec(len(cs), 2, width_ratios=[1.05, 1], hspace=0.7, wspace=0.42,
                          left=0.12, right=0.97, top=top_frac(fig, 1.35), bottom=0.55 / fig.get_figheight())
    role = {"SP": "starter", "RP": "reliever"}
    for i, (_, r) in enumerate(cs.iterrows()):
        a = fig.add_subplot(gs[i, 0])
        style(a, "x")
        parts = [("kbb", r["kbb_pts"]), ("contact", r["contact_pts"]), ("conv", r["conv_pts"])]
        yy = np.arange(4)[::-1]
        vals = [v for _, v in parts] + [r["total_pts"]]
        a.barh(yy, vals, height=0.55, color=[CH_COLORS[c] for c, _ in parts] + [INK], zorder=2)
        a.plot([r["total_lo"], r["total_hi"]], [yy[-1], yy[-1]], color=MUTED, lw=1.2, zorder=3)
        lim = max(4, np.abs(vals).max() * 1.5, abs(r["total_lo"]) * 1.35, abs(r["total_hi"]) * 1.35)
        for y_, v_ in zip(yy, vals):
            end = v_ if y_ != yy[-1] else (r["total_hi"] if v_ >= 0 else r["total_lo"])
            a.text(end + (0.02 * lim if v_ >= 0 else -0.02 * lim), y_, f"{v_:+.1f}", va="center",
                   ha="left" if v_ >= 0 else "right", fontsize=8.5, color=INK2)
        a.axvline(0, color=AXIS, lw=1.0)
        a.set_yticks(yy)
        a.set_yticklabels([CH_LABELS[c] for c, _ in parts] + ["Total (95% interval)"], fontsize=9)
        a.set_xlim(-lim, lim)
        a.set_xlabel("Expected park distortion of wOBA allowed (points)", fontsize=9)
        a.set_title(f"{r['case']}: {r['name']} ({r['team']} {role[r['role']]}, {int(r['bf'])} BF). wOBA "
                    f"{r['woba']:.3f} → park-neutral {r['park_neutral_woba']:.3f}", loc="left", fontsize=10,
                    color=INK)
        b = fig.add_subplot(gs[i, 1])
        style(b, "x")
        e = ex.loc[r["pitcher"]].drop(labels=["name"]).astype(float)
        e = e[e > 0].sort_values(ascending=False).head(7)[::-1]
        b.barh(range(len(e)), 100 * e.to_numpy(), height=0.55, color=DEEMPH, zorder=2)
        for y_, (park, sh) in enumerate(e.items()):
            b.text(100 * sh + 0.8, y_, f"{100 * sh:.0f}%   park {tot.get(park, np.nan):+.0f}", va="center",
                   fontsize=8.5, color=INK2)
        b.set_yticks(range(len(e)))
        b.set_yticklabels([("Other (neutral sites)" if p == "Other" else p) for p in e.index], fontsize=9)
        b.set_xlim(0, 100 * e.max() * 1.65)
        b.set_xlabel("Share of the pitcher's 2026 plate appearances; park = its total effect (points)", fontsize=9)
    handles = [Patch(color=CH_COLORS[c], label=CH_LABELS[c]) for c in CH_COLORS] + [Patch(color=INK, label="Total")]
    fig.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(0.97, top_frac(fig, 0.82)))
    titles(fig, "Three pitchers, three ways parks bend the numbers",
           "Expected distortion = exposure-weighted park channel effects (home games evaluated with the home-team "
           "adjustment). Positive = the parks made the pitcher look worse")
    save(fig, "fig6_case_studies.png")


# --------------------------------------------------------------------------------------------
# 7. Actual vs park-neutral
# --------------------------------------------------------------------------------------------
def fig_actual_vs_neutral():
    t = pd.read_csv(OUT_TABLES / "pitcher_distortion.csv")
    q = t[t["qualifies"]].copy()
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.32, left=0.055, right=0.985,
                          top=top_frac(fig, 1.1), bottom=0.08)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    style(a1, "both")
    lo = q[["woba", "park_neutral_woba"]].min().min() - 0.01
    hi = q[["woba", "park_neutral_woba"]].max().max() + 0.01
    a1.plot([lo, hi], [lo, hi], color=AXIS, lw=1.2, zorder=1)
    for rl, (lab, col) in ROLE.items():
        s = q[q["role"] == rl]
        a1.scatter(s["woba"], s["park_neutral_woba"], s=18, color=col, edgecolor=SURFACE, linewidth=0.7,
                   alpha=0.9, zorder=2, label=lab)
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_aspect("equal")
    big = pd.concat([q.nlargest(4, "total_pts"), q.nsmallest(4, "total_pts")])
    place_labels(a1, big["woba"], big["park_neutral_woba"], big["name"], q["woba"], q["park_neutral_woba"],
                 fontsize=7.5)
    a1.set_xlabel("Actual wOBA allowed, 2026")
    a1.set_ylabel("Park-neutral wOBA allowed")
    a1.legend(loc="upper left")
    panel_title(a1, "All qualifying pitchers; below the diagonal = parks made them look worse")

    d = pd.concat([q.nlargest(12, "total_pts"), q.nsmallest(12, "total_pts")]).sort_values("total_pts")
    style(a2, "x")
    for i, (_, r) in enumerate(d.iterrows()):
        col = ROLE[r["role"]][1]
        a2.plot([r["woba"], r["park_neutral_woba"]], [i, i], color=AXIS, lw=1.6, zorder=1)
        a2.scatter(r["woba"], i, s=26, color=SURFACE, edgecolor=col, linewidth=1.3, zorder=2)
        a2.scatter(r["park_neutral_woba"], i, s=30, color=col, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    a2.axhline(11.5, color=GRID, lw=0.8)
    a2.set_yticks(range(len(d)))
    a2.set_yticklabels([f"{n} ({tm})  {p:+.0f}" for n, tm, p in zip(d["name"], d["team"], d["total_pts"])], fontsize=8)
    a2.set_ylim(-0.7, len(d) - 0.3)
    a2.set_xlabel("wOBA allowed: hollow = actual, filled = park-neutral")
    panel_title(a2, "12 largest distortions each way (points); color = role")
    titles(fig, f"Parks move qualifying pitchers' wOBA by up to {q['total_pts'].abs().max():.0f} points: Rockies "
                "and Athletics look better once neutralized, Mariners and Rangers worse",
           f"Park-neutral wOBA = actual wOBA minus the exposure-weighted park effect (pooled 2024-2026 estimates); "
           f"SD of the adjustment {q['total_pts'].std():.1f} points across {len(q)} pitchers")
    save(fig, "fig7_actual_vs_neutral.png")


# --------------------------------------------------------------------------------------------
# 8. Robustness: pooled vs 2026-only
# --------------------------------------------------------------------------------------------
def fig_pooled_vs_2026():
    d = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    d = d[d["park_key"] != OTHER_PARK]
    w = d.pivot(index="venue_short", columns="fit", values="total_pts").dropna()
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=top_frac(fig, 1.1), bottom=0.08)
    style(ax, "both")
    lo, hi = w.min().min() - 5, w.max().max() + 5
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.2)
    ax.scatter(w["pooled"], w["2026"], s=32, color=BLUE, edgecolor=SURFACE, linewidth=1.1, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    diff = (w["2026"] - w["pooled"]).abs()
    pick = list(set(diff.nlargest(5).index) | {"Coors Field", "T-Mobile Park"})
    place_labels(ax, w.loc[pick, "pooled"], w.loc[pick, "2026"], pick, w["pooled"], w["2026"])
    ax.set_xlabel("Total park effect, 2024-2026 pooled (wOBA points)")
    ax.set_ylabel("Total park effect, 2026 only (wOBA points)")
    r = np.corrcoef(w["pooled"], w["2026"])[0, 1]
    titles(fig, f"One season recovers the park ordering only roughly (r = {r:.2f})",
           "Total effect per park. 2026 is a third of the pooled data, so the two are not independent")
    save(fig, "fig8_pooled_vs_2026.png")


def main():
    for f in (fig_park_channels, fig_pitch_movement, fig_carry, fig_spray, fig_savant, fig_case_studies,
              fig_actual_vs_neutral, fig_pooled_vs_2026):
        f()


if __name__ == "__main__":
    main()
