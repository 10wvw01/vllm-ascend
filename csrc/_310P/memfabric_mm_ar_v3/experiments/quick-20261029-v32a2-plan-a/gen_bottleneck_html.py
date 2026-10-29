#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a self-contained SVG-based bottleneck analysis HTML from the V3.2 A2 run."""
import json
import math
from pathlib import Path

RESULTS = Path("/tmp/opencode/310p-v32a2-20261029-001/results")
OUT = Path(
    "/tmp/opencode/310p-v32a2-20261029-001/vllm-ascend/csrc/_310P/memfabric_mm_ar_v3/"
    "experiments/quick-20261029-v32a2-plan-a/bottleneck_analysis.html"
)


def pct(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[f] * (c - k) + s[c] * (k - f)


def load(m, g, r):
    d = {}
    for rank in (0, 1):
        d[rank] = json.loads((RESULTS / f"timing_r{r}_m{m}_g{g}_rank{rank}.json").read_text())["samples"]
    n = min(len(d[0]), len(d[1]))
    ev_max = [max(d[0][i]["event_us"], d[1][i]["event_us"]) for i in range(n)]
    gaps = [
        max(d[0][i]["wall_us"], d[1][i]["wall_us"]) - max(d[0][i]["event_us"], d[1][i]["event_us"])
        for i in range(n)
    ]
    return ev_max, gaps


def best_p50(m, g):
    ps = [pct(load(m, g, r)[0], 0.5) for r in (0, 1)]
    return min(ps)


# ---------------- data ----------------
STOCK = {2048: 1078.412, 4096: 2022.553}
V2D4 = {2048: 1367.878, 4096: 2539.831}
V31 = {(2048, 512): 1878.913, (2048, 1024): 1655.950, (2048, 2048): 1670.085,
       (4096, 512): 3136.146, (4096, 1024): 2560.319, (4096, 2048): 2516.068}
V32 = {(m, g): best_p50(m, g) for m in (2048, 4096) for g in (512, 1024, 2048)}
PY_GAP = {m: pct(load(m, 1024, 1)[1], 0.5) for m in (2048, 4096)}  # wall-event p50, g1024
IFACE = {m: V32[(m, 1024)] - V31[(m, 1024)] for m in (2048, 4096)}
PIPE = {m: V31[(m, 1024)] for m in (2048, 4096)}
QUANT = {}
for m in (2048, 4096):
    for r in (0, 1):
        ev, _ = load(m, 1024, r)
        QUANT[(m, r)] = (pct(ev, 0.5), pct(ev, 0.9), pct(ev, 0.99))

C = {
    "stock": "#10b981", "v2": "#6366f1", "v31": "#f59e0b", "v32": "#ef4444",
    "py": "#94a3b8", "iface": "#f97316", "pipe": "#3b82f6", "axis": "#475569",
    "grid": "#e2e8f0", "text": "#1e293b", "sub": "#64748b", "mm": "#3b82f6",
    "reduce": "#f97316", "gather": "#a855f7", "finish": "#ef4444", "idle": "#f1f5f9",
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="-apple-system,\'Noto Sans SC\',\'PingFang SC\',\'Microsoft YaHei\',sans-serif">'
        ]

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, rx=4, opacity=1.0, stroke=None):
        st = f'stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" opacity="{opacity}" {st}/>')

    def text(self, x, y, s, size=13, fill=None, anchor="middle", weight="normal"):
        fill = fill or C["text"]
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke, dash=None, width=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def html(self):
        self.add("</svg>")
        return "\n".join(self.parts)


def axis_frame(s, x, y, w, h, ymax, yticks, ylab):
    s.rect(x, y, w, h, "#ffffff", rx=6, stroke=C["grid"])
    for tv in yticks:
        gy = y + h - tv / ymax * h
        s.line(x, gy, x + w, gy, C["grid"], dash="4 4", width=1)
        s.text(x - 8, gy + 4, f"{tv:,.0f}", size=11, fill=C["sub"], anchor="end")
    s.text(x - 44, y + h / 2, ylab, size=12, fill=C["sub"], anchor="middle")
    s.add(f'<g transform="rotate(-90 {x - 44} {y + h / 2})"></g>')


def fmt_v(v):
    return f"{v:,.0f}"


# ---------------- chart 1: overview ----------------
def chart1():
    s = Svg(960, 420)
    x0, y0, w, h, ymax = 90, 60, 840, 280, 3600
    axis_frame(s, x0, y0, w, h, ymax, list(range(0, 3601, 600)), "µs")
    series = [("真实NZ Stock", "stock", STOCK), ("V2 D4", "v2", V2D4),
              ("V3.1最佳", "v31", {m: min(V31[(m, g)] for g in (512, 1024, 2048)) for m in (2048, 4096)}),
              ("V3.2-A2最佳", "v32", {m: min(V32[(m, g)] for g in (512, 1024, 2048)) for m in (2048, 4096)})]
    bw, gap = 70, 14
    for gi, m in enumerate((2048, 4096)):
        gx = x0 + 90 + gi * (w / 2)
        s.text(gx + (len(series) * (bw + gap)) / 2 - gap / 2, y0 + h + 24, f"M={m}（输出{m * 4 // 1024}MiB/Rank）", size=14, weight="bold")
        for si, (name, key, data) in enumerate(series):
            v = data[m]
            bx = gx + si * (bw + gap)
            bh = v / ymax * h
            s.rect(bx, y0 + h - bh, bw, bh, C[key])
            s.text(bx + bw / 2, y0 + h - bh - 22, fmt_v(v), size=12, weight="bold")
            vs = f"+{v / STOCK[m] * 100 - 100:.0f}%"
            s.text(bx + bw / 2, y0 + h - bh - 7, vs, size=11,
                   fill=C["sub"] if key == "stock" else C[key])
    lx = x0 + 10
    for name, key, _ in series:
        s.rect(lx, 18, 14, 14, C[key])
        s.text(lx + 20, 30, name, size=12, anchor="start")
        lx += 20 + len(name) * 13 + 26
    s.text(480, 400, "柱上百分比 = 相对同M真实NZ Stock的时延增加；跨批方向性对照，非同批因果收益", size=11, fill=C["sub"])
    return s.html()


# ---------------- chart 2: stacked layers ----------------
def chart2():
    s = Svg(960, 470)
    x0, y0, w, h, ymax = 110, 70, 700, 300, 3600
    axis_frame(s, x0, y0, w, h, ymax, list(range(0, 3601, 600)), "µs")
    bw = 150
    for gi, m in enumerate((2048, 4096)):
        gx = x0 + 150 + gi * 330
        layers = [("① Python/ctypes\n+同步返回", "py", PY_GAP[m]),
                  ("② A2池接口\n(acquire同步+双Host协商\n+Storage包装)", "iface", IFACE[m]),
                  ("③ 内核+通信管线\n(=V3.1口径)", "pipe", PIPE[m])]
        cy = y0 + h
        for name, key, v in layers:
            bh = v / ymax * h
            cy -= bh
            s.rect(gx, cy, bw, bh, C[key])
            mid = cy + bh / 2
            if bh > 26:
                s.text(gx + bw / 2, mid + 4, f"{v:,.0f}", size=13, fill="#ffffff", weight="bold")
        total = PY_GAP[m] + IFACE[m] + PIPE[m]
        s.text(gx + bw / 2, cy - 26, f"V3.2总计 {total:,.0f}", size=13, weight="bold")
        s.text(gx + bw / 2, cy - 10, f"(实测p50 {V32[(m, 1024)]:,.1f})", size=11, fill=C["sub"])
        s.text(gx + bw / 2, y0 + h + 22, f"M={m} chunk1024", size=14, weight="bold")
        # reference lines
        for rv, lab, col in ((STOCK[m], "Stock", "stock"), (V31[(m, 1024)], "V3.1", "v31")):
            ry = y0 + h - rv / ymax * h
            s.line(x0 + 4, ry, x0 + w, ry, C[col], dash="7 5", width=2)
            s.text(x0 + w - 6, ry - 6, f"{lab} {rv:,.0f}", size=12, fill=C[col], anchor="end")
    ly = 26
    for name, key in [("① Python/ctypes/同步返回(wall−event)", "py"),
                      ("② A2池接口增量(V3.2−V3.1跨批)", "iface"),
                      ("③ 内核+通信管线(V3.1口径)", "pipe")]:
        s.rect(x0 + 10, ly, 14, 14, C[key])
        s.text(x0 + 30, ly + 12, name, size=12, anchor="start")
        ly += 22
    s.text(480, 452, "②为跨批估算(各配置区间M2048:170~420µs / M4096:170~240µs，受批间噪声±10%影响)；③即V3.1同法测量值", size=11, fill=C["sub"])
    return s.html()


# ---------------- chart 3: chunk scaling ----------------
def chart3():
    s = Svg(960, 430)
    x0, y0, w, h, ymax = 90, 70, 840, 280, 3600
    axis_frame(s, x0, y0, w, h, ymax, list(range(0, 3601, 600)), "µs")
    bw = 80
    for gi, m in enumerate((2048, 4096)):
        gx = x0 + 120 + gi * (w / 2)
        s.text(gx + 1.5 * (bw + 30), y0 + h + 46, f"M={m}（总量{m * 4 // 1024}MiB不变）", size=14, weight="bold")
        for ci, g in enumerate((512, 1024, 2048)):
            v = V32[(m, g)]
            bx = gx + ci * (bw + 30)
            bh = v / ymax * h
            s.rect(bx, y0 + h - bh, bw, bh, C["v32"] if g == 1024 else "#fca5a5")
            s.text(bx + bw / 2, y0 + h - bh - 20, fmt_v(v), size=12, weight="bold")
            s.text(bx + bw / 2, y0 + h + 18, f"chunk{g}", size=12)
            s.text(bx + bw / 2, y0 + h + 34, f"{m // g}个chunk", size=11, fill=C["sub"])
        base = min(V32[(m, g)] for g in (512, 1024, 2048))
        by = y0 + h - base / ymax * h
        s.line(gx - 20, by, gx + 3.5 * (bw + 30) - 20, by, C["v31"], dash="7 5", width=2)
        s.text(gx + 1.5 * (bw + 30), by - 8, "同体积下几乎不变", size=12, fill=C["v31"])
    s.text(480, 412, "chunk数从4/8减到1/2，耗时几乎不变（虚线）→ 耗时由总数据量(R/D/G搬运+MM)决定，不是逐chunk控制开销；chunk512的额外~280µs是波数/包数/quiet惩罚", size=11, fill=C["sub"])
    return s.html()


# ---------------- chart 4: pipeline schematic ----------------
def chart4():
    s = Svg(960, 520)
    s.text(480, 28, "当前Kernel流水：Gather与输出搬运被串行进Cube停顿窗口（结构示意，非实测分段）", size=15, weight="bold")
    lane_y, lane_h = 60, 44
    lx, lw = 150, 760

    def lane(x, y, w, label):
        s.rect(x, y, w, lane_h, "#f8fafc", rx=6, stroke=C["grid"])
        s.text(x - 12, y + lane_h / 2 + 4, label, size=12, fill=C["sub"], anchor="end")

    def block(x, y, w, txt, color, tcol="#ffffff"):
        s.rect(x, y + 5, w, lane_h - 10, color, rx=4)
        s.text(x + w / 2, y + lane_h / 2 + 4, txt, size=11.5, fill=tcol, weight="bold")

    # current
    s.text(lx, 52, "当前（V3/V3.1/V3.2同构）：每chunk循环 MM(g) → Finish(g-1) → SubmitReduce(g)", size=13, anchor="start", weight="bold")
    lane(lx, lane_y, lw, "Cube (MM)")
    lane(lx, lane_y + lane_h + 8, lw, "SDMA")
    y1 = lane_y
    y2 = lane_y + lane_h + 8
    seq = [("MM0", 90, "mm"), ("R0", 60, "reduce"), ("Finish0\n收件+quiet+Gather+quiet+输出搬运", 200, "finish"),
           ("MM1", 90, "mm"), ("R1", 60, "reduce"), ("Finish1 …", 160, "finish"), ("MM2", 70, "mm")]
    cx = lx + 8
    prev_bottom = None
    for name, w_, col in seq:
        if col == "mm":
            block(cx, y1, w_, name.replace("\n", " "), C[col])
            if prev_bottom is not None:
                s.line(cx - 4, y2 + lane_h, cx - 4, y1 + lane_h + 6, C["finish"], width=1.5, dash="3 3")
            prev_bottom = True
            cx += w_ + 6
        else:
            block(cx, y2, w_, name.replace("\n", " "), C[col])
            cx += w_ + 6
    s.text(lx + lw, y2 + lane_h + 16, "↑ Gather/输出搬运期间 MM 停等；M2048/chunk2048只有一组，完全没有交叠", size=12, fill=C["finish"], anchor="end")

    # target
    ty = y2 + lane_h + 44
    s.text(lx, ty - 8, "研发目标（V3.1重排合同）：MM(k) ‖ Reduce(k-1) ‖ Gather(k-2) 三段交叠", size=13, anchor="start", weight="bold")
    lane(lx, ty, lw, "Cube (MM)")
    lane(lx, ty + lane_h + 8, lw, "SDMA")
    yy1, yy2 = ty, ty + lane_h + 8
    bx = lx + 8
    for i in range(3):
        block(bx, yy1, 150, f"MM{i}", C["mm"])
        bx += 156
    bx = lx + 8
    for i in range(3):
        block(bx, yy2, 88, f"R{i}", C["reduce"])
        block(bx + 94, yy2, 88, f"G{i}", C["gather"])
        bx += 188
    s.text(lx + lw, yy2 + lane_h + 16, "↑ Reduce/Gather在MM期间在途；已完成的旧tile才拷出（每波最多3个活跃tile）", size=12, fill=C["gather"], anchor="end")

    ly = ty + 2 * lane_h + 66
    for name, col in [("MM（Cube计算）", "mm"), ("Reduce提交", "reduce"), ("Gather", "gather"),
                      ("Finish串行段（Core0收件/quiet/输出搬运）", "finish")]:
        s.rect(lx + 8, ly, 14, 14, C[col])
        s.text(lx + 28, ly + 12, name, size=12, anchor="start")
        ly += 22
    s.text(480, ly + 14, "该串行结构 + 每chunk READY往返/两次quiet + 逐tile输出搬运PIPE_ALL屏障 + Core0单生产者，构成③层相对Stock约540~580µs的管线差距", size=11, fill=C["sub"])
    return s.html()


# ---------------- chart 5: jitter ----------------
def chart5():
    s = Svg(960, 330)
    x0, y0, w, h, ymax = 90, 60, 840, 200, 3600
    axis_frame(s, x0, y0, w, h, ymax, list(range(0, 3601, 900)), "µs")
    bw, gap = 46, 10
    labels = [("M2048 轮1", (2048, 0)), ("M2048 轮2", (2048, 1)), ("M4096 轮1", (4096, 0)), ("M4096 轮2", (4096, 1))]
    for gi, (lab, key) in enumerate(labels):
        gx = x0 + 120 + gi * 190
        p50, p90, p99 = QUANT[key]
        for qi, (v, col, name) in enumerate(((p50, C["v32"], "p50"), (p90, "#fca5a5", "p90"), (p99, "#fecaca", "p99"))):
            bx = gx + qi * (bw + gap)
            bh = v / ymax * h
            s.rect(bx, y0 + h - bh, bw, bh, col)
            if qi == 0:
                s.text(bx + bw / 2, y0 + h - bh - 20, fmt_v(v), size=12, weight="bold")
        s.text(gx + 1.5 * bw + gap, y0 + h + 20, lab, size=13, weight="bold")
        s.text(gx + 1.5 * bw + gap, y0 + h + 36, f"p99/p50 = {p99 / p50:.3f}", size=11, fill=C["sub"])
    ly = 26
    for name, col in (("p50", C["v32"]), ("p90", "#fca5a5"), ("p99", "#fecaca")):
        s.rect(x0 + 200 + (0 if name == "p50" else (150 if name == "p90" else 300)), ly, 14, 14, col)
        s.text(x0 + 216 + (0 if name == "p50" else (150 if name == "p90" else 300)), ly + 12, name, size=12, anchor="start")
    s.text(480, 312, "Rank不对称<1%、p99/p50≤1.09 → 慢不在抖动/尾延迟/Host协商波动，而在每拍的固定结构成本", size=11, fill=C["sub"])
    return s.html()


# ---------------- assemble ----------------
css = """
body{font-family:-apple-system,'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif;
     max-width:1000px;margin:24px auto;padding:0 16px;color:#1e293b;background:#f8fafc}
h1{font-size:22px;border-bottom:3px solid #ef4444;padding-bottom:8px}
h2{font-size:17px;margin-top:34px;background:#eef2ff;padding:8px 12px;border-left:5px solid #6366f1;border-radius:4px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-top:10px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.tldr{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 18px;margin-top:12px}
.tldr li{margin:6px 0}
.note{font-size:12px;color:#64748b;margin-top:6px}
svg{width:100%;height:auto;display:block}
b.r{color:#ef4444}b.g{color:#10b981}
"""
html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>V3.2-A2 性能瓶颈可视化分析（2026-10-29）</title><style>{css}</style></head><body>
<h1>V3.2-A2 为什么慢：性能瓶颈可视化分析</h1>
<div class="note">数据来源：本轮六配置两轮实测（双Rank max后p50，完整接口窗口）＋历史批次对照。
未开独立归因轮/Profiler（超授权），分层②为跨批估算。</div>

<div class="tldr"><b>三句话结论</b>
<ol>
<li><b>最慢的大头是内核+通信管线本身</b>（占82%~94%）：Gather/输出搬运被串行进Cube停顿窗口、每chunk READY往返+两次quiet、逐tile PIPE_ALL屏障——这套结构相对Stock差约<b class="r">540~580µs</b>。</li>
<li><b>A2接口是第二瓶颈</b>：每调用多付<b class="r">~170~420µs</b>（acquire流同步+两次Host协商+Storage包装），这是买"输出生命周期安全"的价格，M2048时占总耗时18%。</li>
<li><b>不是抖动问题</b>：Rank不对称&lt;1%、p99/p50≤1.09；时间随总数据量走、不随chunk数走（边际~0.35µs/行甚至低于Stock的~0.46），输在固定结构成本，不在边际吞吐。</li>
</ol></div>

<h2>图1｜和 Stock / V2 D4 / V3.1 放在一起看：差距有多大</h2>
<div class="card">{chart1()}</div>
<div class="note">V3.2最佳(M2048:2018.7 / M4096:2727.1)比Stock慢87%/35%，比V2 D4慢48%/7%，比V3.1慢22%/8%。</div>

<h2>图2｜V3.2 的每一拍时间花在哪（分层拆解）</h2>
<div class="card">{chart2()}</div>
<div class="note">一根柱从下往上：①Python/ctypes/同步返回(~90µs) → ②A2池接口增量 → ③内核+通信管线(=V3.1口径)。
②+①就是V3.2比V3.1多付的钱；③与Stock虚线的距离是内核结构差距。</div>

<h2>图3｜chunk大小为什么不重要：耗时跟着总数据量走</h2>
<div class="card">{chart3()}</div>

<h2>图4｜管线结构问题：Gather被串行（当前 vs 研发目标三段交叠）</h2>
<div class="card">{chart4()}</div>

<h2>图5｜排除抖动：p50/p90/p99几乎贴着</h2>
<div class="card">{chart5()}</div>

<h2>瓶颈排序与去向</h2>
<div class="card"><ol>
<li><b>③内核+通信管线（M2048约1.66ms / M4096约2.56ms，占82%~94%）</b>——需按研发侧复核方案重排：MM(k)‖Reduce(k-1)‖Gather(k-2)三段交叠、减少包与quiet、消除逐tile输出搬运串行；属方案/算法侧，交回研发决策。</li>
<li><b>②A2池接口（每调用约170~420µs）</b>——acquire流同步+双Host协商是主成分；除非跨层复用输出槽摊薄，否则A2形态在中小M先天吃亏。</li>
<li><b>①Python/ctypes（~90µs）</b>——占比小，非重点。</li>
</ol>
<div class="note">精确到MM/READY/Reduce/Gather/IB/MTE各自的设备耗时需要插桩归因轮（本轮未授权）；本页全部由既有样本数据+跨批对照推导。</div></div>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"written: {OUT} ({OUT.stat().st_size} bytes)")
print("V32:", {k: round(v, 1) for k, v in V32.items()})
print("PY_GAP:", {k: round(v, 1) for k, v in PY_GAP.items()})
print("IFACE:", {k: round(v, 1) for k, v in IFACE.items()})
