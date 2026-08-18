from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapacityPoint:
    model: str
    source_width: int
    width: int
    retained_ratio: float
    method: str
    protocol: str
    arc: float | None
    gsm8k: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Gemma4 and cross-model MoE capacity curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("capacity_cliff_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("experiments") / "capacity_cliff_v1",
    )
    return parser.parse_args()


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def load_points(path: Path) -> list[CapacityPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = [
        CapacityPoint(
            model=row["model"],
            source_width=int(row["source_width"]),
            width=int(row["width"]),
            retained_ratio=float(row["retained_ratio"]),
            method=row["method"],
            protocol=row["protocol"],
            arc=optional_float(row["arc"]),
            gsm8k=optional_float(row["gsm8k"]),
            status=row["status"],
        )
        for row in rows
    ]
    validate_points(points)
    return points


def validate_points(points: list[CapacityPoint]) -> None:
    if not points:
        raise ValueError("Capacity table is empty")
    keys = [(point.model, point.width) for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError("Capacity table contains duplicate model/width rows")
    for point in points:
        expected_ratio = point.width / point.source_width
        if abs(point.retained_ratio - expected_ratio) > 1.0e-8:
            raise ValueError(f"Incorrect retained ratio for {point.model} width {point.width}")
        has_scores = point.arc is not None and point.gsm8k is not None
        if point.status == "complete" and not has_scores:
            raise ValueError(f"Complete row lacks scores: {point.model} width {point.width}")
        if point.status == "pending" and has_scores:
            raise ValueError(f"Pending row unexpectedly has scores: {point.model} width {point.width}")


def render_svg_fallback(points: list[CapacityPoint], output_path: Path) -> None:
    width, height = 1200, 800
    panel_width, panel_height = 520, 280
    origins = ((80, 80), (650, 80), (80, 440), (650, 440))
    colors = {"Gemma4": "#c23b22", "Qwen3": "#246a73", "Qwen3.6": "#6a4c93"}
    panels = (
        ("Gemma4 ARC", "width", "arc"),
        ("Gemma4 GSM8K", "width", "gsm8k"),
        ("ARC by retained capacity", "ratio", "arc"),
        ("GSM8K by retained capacity", "ratio", "gsm8k"),
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#222}.title{font-size:20px;font-weight:600}.axis{font-size:13px}.pending{font-size:12px;fill:#777}</style>',
        '<text x="600" y="34" text-anchor="middle" class="title">MoE Capacity Cliff: Gemma4 CHANNEL vs Qwen AIMER Baselines</text>',
    ]
    for (title, x_mode, metric), (origin_x, origin_y) in zip(panels, origins):
        plot_left, plot_top = origin_x + 55, origin_y + 35
        plot_width, plot_height = panel_width - 75, panel_height - 75
        lines.extend([
            f'<text x="{origin_x + panel_width / 2}" y="{origin_y + 18}" text-anchor="middle" class="title">{html.escape(title)}</text>',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333"/>',
            f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#333"/>',
        ])
        x_min, x_max = ((352.0, 704.0) if x_mode == "width" else (0.48, 1.02))
        y_min, y_max = 0.65, 1.0
        for tick in (0.7, 0.8, 0.9, 1.0):
            y = plot_top + (y_max - tick) / (y_max - y_min) * plot_height
            lines.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" stroke="#ddd"/>')
            lines.append(f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')
        models = ("Gemma4",) if x_mode == "width" else ("Gemma4", "Qwen3", "Qwen3.6")
        for model in models:
            model_points = sorted(
                (point for point in points if point.model == model and point.status == "complete"),
                key=lambda point: point.width if x_mode == "width" else point.retained_ratio,
            )
            coordinates = []
            for point in model_points:
                x_value = float(point.width) if x_mode == "width" else point.retained_ratio
                y_value = getattr(point, metric)
                if y_value is None:
                    continue
                x = plot_left + (x_value - x_min) / (x_max - x_min) * plot_width
                y = plot_top + (y_max - y_value) / (y_max - y_min) * plot_height
                coordinates.append((x, y))
            if coordinates:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
                lines.append(f'<polyline points="{path}" fill="none" stroke="{colors[model]}" stroke-width="3"/>')
                for x, y in coordinates:
                    lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[model]}"/>')
        if x_mode == "width":
            for point in points:
                if point.model != "Gemma4" or point.status != "pending":
                    continue
                x = plot_left + (point.width - x_min) / (x_max - x_min) * plot_width
                lines.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_top + plot_height}" stroke="#999" stroke-dasharray="4 4"/>')
                lines.append(f'<text x="{x - 4:.1f}" y="{plot_top + 15}" text-anchor="end" class="pending" transform="rotate(-90 {x - 4:.1f} {plot_top + 15})">pending {point.width}</text>')
        x_label = "Physical expert width" if x_mode == "width" else "Retained expert width (%)"
        lines.append(f'<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 38}" text-anchor="middle" class="axis">{x_label}</text>')
    legend_x = 930
    for index, model in enumerate(("Gemma4", "Qwen3", "Qwen3.6")):
        y = 755
        x = legend_x + index * 85
        lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[model]}"/><text x="{x + 10}" y="{y + 4}" class="axis">{model}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(points: list[CapacityPoint], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        output_dir.mkdir(parents=True, exist_ok=True)
        render_svg_fallback(points, output_dir / "capacity_cliff.svg")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Gemma4": "#c23b22", "Qwen3": "#246a73", "Qwen3.6": "#6a4c93"}
    markers = {"Gemma4": "o", "Qwen3": "s", "Qwen3.6": "^"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    gemma = sorted((point for point in points if point.model == "Gemma4"), key=lambda point: point.width)
    complete_gemma = [point for point in gemma if point.status == "complete"]
    pending_gemma = [point for point in gemma if point.status == "pending"]

    for axis, metric, title in zip(axes[0], ("arc", "gsm8k"), ("Gemma4 ARC", "Gemma4 GSM8K")):
        axis.plot(
            [point.width for point in complete_gemma],
            [getattr(point, metric) for point in complete_gemma],
            color=colors["Gemma4"],
            marker=markers["Gemma4"],
            linewidth=2,
        )
        for point in pending_gemma:
            axis.axvline(point.width, color="#999999", linestyle=":", linewidth=1)
            axis.text(point.width, 0.68, "pending", rotation=90, ha="right", va="bottom", fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Physical expert width")
        axis.set_ylabel("Accuracy")
        axis.set_xticks([point.width for point in gemma])
        axis.set_ylim(0.65, 1.0)
        axis.grid(alpha=0.25)

    complete = [point for point in points if point.status == "complete"]
    for axis, metric, title in zip(
        axes[1],
        ("arc", "gsm8k"),
        ("ARC by retained capacity", "GSM8K by retained capacity"),
    ):
        for model in ("Gemma4", "Qwen3", "Qwen3.6"):
            model_points = sorted(
                (point for point in complete if point.model == model),
                key=lambda point: point.retained_ratio,
            )
            axis.plot(
                [100.0 * point.retained_ratio for point in model_points],
                [getattr(point, metric) for point in model_points],
                color=colors[model],
                marker=markers[model],
                linewidth=2,
                label=model,
            )
        axis.set_title(title)
        axis.set_xlabel("Retained expert width (%)")
        axis.set_ylabel("Accuracy")
        axis.set_xlim(48, 102)
        axis.set_ylim(0.65, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle("MoE Capacity Cliff: Gemma4 CHANNEL vs Qwen AIMER Baselines", fontsize=14)
    figure.savefig(output_dir / "capacity_cliff.png", dpi=180)
    figure.savefig(output_dir / "capacity_cliff.svg")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    points = load_points(args.input.expanduser().resolve())
    plot(points, args.output_dir.expanduser().resolve())
    complete = sum(point.status == "complete" for point in points)
    pending = sum(point.status == "pending" for point in points)
    print(f"Rendered {complete} complete points; {pending} pending points remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())