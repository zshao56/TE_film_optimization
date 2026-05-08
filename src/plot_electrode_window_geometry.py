from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def line(x1, y1, x2, y2, stroke="#333", width=1.2, dash=None, marker_end=False):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if marker_end else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{marker}/>'
    )


def rect(x, y, w, h, fill, stroke="none", width=1.0, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
    )


def text(x, y, value, size=15, fill="#222", anchor="middle", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{value}</text>'
    )


def double_arrow(x1, y1, x2, y2, label, lx, ly, color="#333"):
    return "\n".join(
        [
            line(x1, y1, x2, y2, color, 1.2, marker_end=True),
            line(x2, y2, x1, y1, color, 1.2, marker_end=True),
            text(lx, ly, label, 14, color),
        ]
    )


def generate_svg():
    # Drawing coordinates are in pixels. Geometry uses a normalized L/h ratio.
    sx = 74.0
    sy = 74.0
    x0 = 110.0
    y0 = 255.0
    length_L = 10.0
    thickness_h = 2.0
    w_m = 1.0
    t_m = 0.28
    s_min = 1.0
    x_cold = 2.3
    x_hot = 7.4

    width = 980
    height = 455
    domain_w = length_L * sx
    domain_h = thickness_h * sy
    top_y = y0 - domain_h
    band_h = t_m * sy
    band_y = top_y

    def px(x):
        return x0 + x * sx

    def py(z):
        return y0 - z * sy

    cold_left = px(x_cold - w_m / 2)
    cold_right = px(x_cold + w_m / 2)
    hot_left = px(x_hot - w_m / 2)
    hot_right = px(x_hot + w_m / 2)
    cold_center = px(x_cold)
    hot_center = px(x_hot)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="context-stroke"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, width, height, "#ffffff"),
        text(width / 2, 36, "Top electrode-window search geometry", 22, "#222", weight="700"),
        text(width / 2, 62, "Search fixed-size top windows; maximize averaged electrode temperature difference", 15, "#333"),
        rect(x0, top_y, domain_w, domain_h, "#f7f3ea", "#222", 1.8),
        rect(x0, top_y, domain_w, domain_h, "#d7e8f7", "none", 1, 0.42),
        f'<polyline points="{px(0):.1f},{py(0.15):.1f} {px(1.4):.1f},{py(1.65):.1f} {px(2.7):.1f},{py(0.95):.1f} {px(4.4):.1f},{py(1.8):.1f} {px(6.1):.1f},{py(0.45):.1f} {px(7.8):.1f},{py(1.35):.1f} {px(10):.1f},{py(0.25):.1f}" fill="none" stroke="#44546a" stroke-width="2"/>',
        text(px(5.0), py(0.52), "two-material substrate geometry", 14, "#44546a"),
        rect(x0, band_y, domain_w, band_h, "#f1c232", "none", 1, 0.24),
        text(x0 + domain_w + 20, band_y + band_h / 2 + 5, "top sampling band", 14, "#7a5a00", "start"),
        rect(cold_left, band_y, w_m * sx, band_h, "#4f81bd", "#1f4e79", 1.6, 0.9),
        rect(hot_left, band_y, w_m * sx, band_h, "#c0504d", "#7f1d1d", 1.6, 0.92),
        text(cold_center, top_y - 26, "cold electrode", 14, "#1f4e79"),
        text(cold_center, top_y - 9, "window", 14, "#1f4e79"),
        text(hot_center, top_y - 26, "hot electrode", 14, "#7f1d1d"),
        text(hot_center, top_y - 9, "window", 14, "#7f1d1d"),
        line(cold_center, band_y, cold_center, band_y + band_h + 22, "#1f4e79", 1.2, "5,4"),
        line(hot_center, band_y, hot_center, band_y + band_h + 22, "#7f1d1d", 1.2, "5,4"),
        text(cold_center, band_y + band_h + 38, "x_cold_electrode", 13, "#1f4e79"),
        text(hot_center, band_y + band_h + 38, "x_hot_electrode", 13, "#7f1d1d"),
        double_arrow(x0, y0 + 46, x0 + domain_w, y0 + 46, "L", x0 + domain_w / 2, y0 + 74),
        double_arrow(x0 - 45, y0, x0 - 45, top_y, "h", x0 - 72, top_y + domain_h / 2 + 5),
        double_arrow(hot_left, top_y - 48, hot_right, top_y - 48, "w_m", hot_center, top_y - 62, "#7f1d1d"),
        double_arrow(x0 + domain_w + 18, band_y, x0 + domain_w + 18, band_y + band_h, "t_m", x0 + domain_w + 52, band_y + band_h / 2 + 5, "#7a5a00"),
        double_arrow(cold_right, band_y + band_h + 58, hot_left, band_y + band_h + 58, "gap >= w_m + s_min", (cold_right + hot_left) / 2, band_y + band_h + 84, "#666"),
    ]

    for x in [1.2, 3.6, 6.0, 8.4]:
        parts.append(line(px(x), y0 + 26, px(x), y0 + 2, "#b45f06", 1.4, marker_end=True))
    parts.append(text(px(5), y0 + 105, "bottom thermal source or sink", 14, "#b45f06"))

    for x in [0.9, 2.6, 4.3, 6.0, 7.7, 9.4]:
        parts.append(line(px(x), top_y - 74, px(x), top_y - 18, "#38761d", 1.4, marker_end=True))
    parts.append(text(px(5), top_y - 91, "top boundary: convection/contact layer, not whole-surface fixed T", 14, "#38761d"))

    parts.extend(
        [
            text(px(5), y0 + 135, "Delta T_parallel = T_hot_electrode_avg - T_cold_electrode_avg", 16, "#222"),
            text(px(5), y0 + 160, "The colored windows are example search results, not fixed end positions.", 14, "#555"),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main():
    svg_path = RESULTS_DIR / "electrode_window_geometry.svg"
    svg_path.write_text(generate_svg(), encoding="utf-8")
    print(svg_path)


if __name__ == "__main__":
    main()
