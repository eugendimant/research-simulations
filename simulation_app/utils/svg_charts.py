"""
Pure SVG-based chart generators that work WITHOUT matplotlib.
These are guaranteed fallbacks that produce visualization no matter what.

All functions return SVG strings that can be embedded directly in HTML.
"""

from typing import Dict, List, Tuple, Optional
import math


# Color palette (colorblind-friendly)
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#95a5a6']


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def create_bar_chart_svg(
    data: Dict[str, Tuple[float, float]],
    title: str = "Comparison by Condition",
    ylabel: str = "Mean Score",
    effect_size: Optional[float] = None,
    p_value: Optional[float] = None,
    width: int = 600,
    height: int = 400
) -> str:
    """
    Create an SVG bar chart with error bars.

    Args:
        data: Dict mapping condition names to (mean, std_error) tuples
        title: Chart title
        ylabel: Y-axis label
        effect_size: Optional Cohen's d value to display
        p_value: Optional p-value to display
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        SVG string that can be embedded in HTML
    """
    if not data:
        return _create_no_data_svg(width, height, "No data available for bar chart")

    # Chart dimensions
    margin_left = 70
    margin_right = 30
    margin_top = 60
    margin_bottom = 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    conditions = list(data.keys())
    means = [data[c][0] for c in conditions]
    errors = [data[c][1] for c in conditions]

    n_bars = len(conditions)
    if n_bars == 0:
        return _create_no_data_svg(width, height, "No conditions to display")

    # Calculate scale
    max_val = max(m + e for m, e in zip(means, errors)) if means else 1
    min_val = min(m - e for m, e in zip(means, errors)) if means else 0

    # Add padding to y-axis range
    y_range = max_val - min_val if max_val != min_val else 1
    y_min = max(0, min_val - y_range * 0.1)
    y_max = max_val + y_range * 0.2

    # Bar layout
    bar_width = min(80, chart_width / n_bars * 0.7)
    bar_spacing = chart_width / n_bars

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        # Background
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # Title
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">{_escape_xml(title)}</text>',
    ]

    # Y-axis label (rotated)
    svg_parts.append(
        f'<text x="20" y="{margin_top + chart_height/2}" text-anchor="middle" font-size="12" fill="#2c3e50" transform="rotate(-90 20 {margin_top + chart_height/2})">{_escape_xml(ylabel)}</text>'
    )

    # Draw grid lines and y-axis labels
    n_grid = 5
    for i in range(n_grid + 1):
        y_val = y_min + (y_max - y_min) * i / n_grid
        y_pos = margin_top + chart_height - (y_val - y_min) / (y_max - y_min) * chart_height

        # Grid line
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + chart_width}" y2="{y_pos}" stroke="#ecf0f1" stroke-width="1"/>'
        )
        # Y-axis label
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4}" text-anchor="end" font-size="10" fill="#7f8c8d">{y_val:.2f}</text>'
        )

    # Draw axes
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#bdc3c7" stroke-width="2"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#bdc3c7" stroke-width="2"/>'
    )

    # Draw bars with error bars
    for i, (cond, mean, error) in enumerate(zip(conditions, means, errors)):
        x_center = margin_left + bar_spacing * (i + 0.5)
        x_left = x_center - bar_width / 2

        # Calculate bar height
        bar_top = margin_top + chart_height - (mean - y_min) / (y_max - y_min) * chart_height
        bar_bottom = margin_top + chart_height - (y_min - y_min) / (y_max - y_min) * chart_height
        bar_height = bar_bottom - bar_top

        color = COLORS[i % len(COLORS)]

        # Bar
        svg_parts.append(
            f'<rect x="{x_left}" y="{bar_top}" width="{bar_width}" height="{bar_height}" fill="{color}" opacity="0.85" rx="2"/>'
        )

        # Error bar
        error_top = margin_top + chart_height - (mean + error - y_min) / (y_max - y_min) * chart_height
        error_bottom = margin_top + chart_height - (mean - error - y_min) / (y_max - y_min) * chart_height

        # Vertical line
        svg_parts.append(
            f'<line x1="{x_center}" y1="{error_top}" x2="{x_center}" y2="{error_bottom}" stroke="#2c3e50" stroke-width="2"/>'
        )
        # Top cap
        svg_parts.append(
            f'<line x1="{x_center - 8}" y1="{error_top}" x2="{x_center + 8}" y2="{error_top}" stroke="#2c3e50" stroke-width="2"/>'
        )
        # Bottom cap
        svg_parts.append(
            f'<line x1="{x_center - 8}" y1="{error_bottom}" x2="{x_center + 8}" y2="{error_bottom}" stroke="#2c3e50" stroke-width="2"/>'
        )

        # Value label above bar
        svg_parts.append(
            f'<text x="{x_center}" y="{error_top - 8}" text-anchor="middle" font-size="11" font-weight="bold" fill="#2c3e50">{mean:.2f}</text>'
        )

        # Condition label below (truncate if too long)
        label = cond if len(cond) <= 15 else cond[:12] + "..."
        svg_parts.append(
            f'<text x="{x_center}" y="{margin_top + chart_height + 20}" text-anchor="middle" font-size="10" fill="#2c3e50" transform="rotate(-25 {x_center} {margin_top + chart_height + 20})">{_escape_xml(label)}</text>'
        )

    # Add p-value and effect size annotation
    if p_value is not None or effect_size is not None:
        annotation_y = margin_top + 15
        annotation_parts = []
        if p_value is not None:
            sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
            annotation_parts.append(f"p = {p_value:.4f} {sig}")
        if effect_size is not None:
            annotation_parts.append(f"d = {effect_size:.2f}")

        annotation_text = " | ".join(annotation_parts)

        # Background box
        box_width = len(annotation_text) * 7 + 20
        svg_parts.append(
            f'<rect x="{width - margin_right - box_width}" y="{annotation_y - 12}" width="{box_width}" height="20" fill="#ecf0f1" rx="4"/>'
        )
        svg_parts.append(
            f'<text x="{width - margin_right - 10}" y="{annotation_y}" text-anchor="end" font-size="11" fill="#2c3e50">{annotation_text}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def create_distribution_svg(
    data: Dict[str, List[float]],
    title: str = "Score Distribution by Condition",
    xlabel: str = "Score",
    width: int = 600,
    height: int = 350
) -> str:
    """
    Create an SVG showing distribution as horizontal range plots with min/max/mean/median.

    Args:
        data: Dict mapping condition names to lists of values
        title: Chart title
        xlabel: X-axis label
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    if not data or all(len(v) == 0 for v in data.values()):
        return _create_no_data_svg(width, height, "No data available for distribution plot")

    margin_left = 120
    margin_right = 30
    margin_top = 50
    margin_bottom = 50
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    conditions = list(data.keys())
    n_conditions = len(conditions)

    if n_conditions == 0:
        return _create_no_data_svg(width, height, "No conditions to display")

    # Calculate global min/max for scale
    all_values = [v for values in data.values() for v in values if not math.isnan(v)]
    if not all_values:
        return _create_no_data_svg(width, height, "No valid data values")

    global_min = min(all_values)
    global_max = max(all_values)
    x_range = global_max - global_min if global_max != global_min else 1
    x_min = global_min - x_range * 0.05
    x_max = global_max + x_range * 0.05

    row_height = min(50, chart_height / n_conditions)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">{_escape_xml(title)}</text>',
        f'<text x="{margin_left + chart_width/2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#2c3e50">{_escape_xml(xlabel)}</text>',
    ]

    # X-axis grid and labels
    n_ticks = 5
    for i in range(n_ticks + 1):
        x_val = x_min + (x_max - x_min) * i / n_ticks
        x_pos = margin_left + chart_width * i / n_ticks

        svg_parts.append(
            f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + chart_height}" stroke="#ecf0f1" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x_pos}" y="{margin_top + chart_height + 15}" text-anchor="middle" font-size="10" fill="#7f8c8d">{x_val:.1f}</text>'
        )

    # Draw distribution for each condition
    for i, cond in enumerate(conditions):
        values = [v for v in data[cond] if not math.isnan(v)]
        if not values:
            continue

        y_center = margin_top + row_height * (i + 0.5)
        color = COLORS[i % len(COLORS)]

        # Calculate statistics
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cond_min = sorted_vals[0]
        cond_max = sorted_vals[-1]
        cond_mean = sum(sorted_vals) / n
        cond_median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2

        # Q1 and Q3 for box
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]

        # Convert to x positions
        def to_x(val):
            return margin_left + (val - x_min) / (x_max - x_min) * chart_width

        x_min_pos = to_x(cond_min)
        x_max_pos = to_x(cond_max)
        x_q1 = to_x(q1)
        x_q3 = to_x(q3)
        x_mean = to_x(cond_mean)
        x_median = to_x(cond_median)

        # Condition label
        label = cond if len(cond) <= 15 else cond[:12] + "..."
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y_center + 4}" text-anchor="end" font-size="11" fill="#2c3e50">{_escape_xml(label)}</text>'
        )

        # Whisker line (min to max)
        svg_parts.append(
            f'<line x1="{x_min_pos}" y1="{y_center}" x2="{x_max_pos}" y2="{y_center}" stroke="{color}" stroke-width="2"/>'
        )

        # Min and max caps
        svg_parts.append(
            f'<line x1="{x_min_pos}" y1="{y_center - 8}" x2="{x_min_pos}" y2="{y_center + 8}" stroke="{color}" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<line x1="{x_max_pos}" y1="{y_center - 8}" x2="{x_max_pos}" y2="{y_center + 8}" stroke="{color}" stroke-width="2"/>'
        )

        # IQR box
        box_height = 20
        svg_parts.append(
            f'<rect x="{x_q1}" y="{y_center - box_height/2}" width="{x_q3 - x_q1}" height="{box_height}" fill="{color}" opacity="0.3" stroke="{color}" stroke-width="2"/>'
        )

        # Median line
        svg_parts.append(
            f'<line x1="{x_median}" y1="{y_center - box_height/2}" x2="{x_median}" y2="{y_center + box_height/2}" stroke="{color}" stroke-width="3"/>'
        )

        # Mean diamond
        svg_parts.append(
            f'<polygon points="{x_mean},{y_center - 6} {x_mean + 5},{y_center} {x_mean},{y_center + 6} {x_mean - 5},{y_center}" fill="#2c3e50"/>'
        )

    # Legend
    svg_parts.append(f'<text x="{width - 100}" y="{margin_top + 10}" font-size="9" fill="#7f8c8d">◆ Mean  | Median</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def create_histogram_svg(
    data: Dict[str, List[float]],
    title: str = "Score Distribution Histogram",
    xlabel: str = "Score",
    n_bins: int = 10,
    width: int = 600,
    height: int = 350
) -> str:
    """
    Create an SVG histogram showing overlapping distributions.

    Args:
        data: Dict mapping condition names to lists of values
        title: Chart title
        xlabel: X-axis label
        n_bins: Number of histogram bins
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    if not data or all(len(v) == 0 for v in data.values()):
        return _create_no_data_svg(width, height, "No data available for histogram")

    margin_left = 60
    margin_right = 120  # Space for legend
    margin_top = 50
    margin_bottom = 60
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    # Calculate global range
    all_values = [v for values in data.values() for v in values if not math.isnan(v)]
    if not all_values:
        return _create_no_data_svg(width, height, "No valid data values")

    global_min = min(all_values)
    global_max = max(all_values)
    bin_width = (global_max - global_min) / n_bins if global_max != global_min else 1

    # Calculate bins for each condition
    condition_bins = {}
    max_count = 0
    for cond, values in data.items():
        valid_values = [v for v in values if not math.isnan(v)]
        bins = [0] * n_bins
        for v in valid_values:
            bin_idx = min(int((v - global_min) / bin_width), n_bins - 1)
            bins[bin_idx] += 1
        condition_bins[cond] = bins
        max_count = max(max_count, max(bins) if bins else 0)

    if max_count == 0:
        return _create_no_data_svg(width, height, "No data to display in histogram")

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{(margin_left + width - margin_right)/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">{_escape_xml(title)}</text>',
        f'<text x="{(margin_left + width - margin_right)/2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#2c3e50">{_escape_xml(xlabel)}</text>',
        f'<text x="20" y="{margin_top + chart_height/2}" text-anchor="middle" font-size="12" fill="#2c3e50" transform="rotate(-90 20 {margin_top + chart_height/2})">Count</text>',
    ]

    # Y-axis grid
    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        y_val = max_count * i / n_y_ticks
        y_pos = margin_top + chart_height - chart_height * i / n_y_ticks

        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + chart_width}" y2="{y_pos}" stroke="#ecf0f1" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 8}" y="{y_pos + 4}" text-anchor="end" font-size="10" fill="#7f8c8d">{int(y_val)}</text>'
        )

    # X-axis labels
    for i in range(n_bins + 1):
        x_val = global_min + bin_width * i
        x_pos = margin_left + chart_width * i / n_bins

        if i % 2 == 0:  # Show every other label to avoid crowding
            svg_parts.append(
                f'<text x="{x_pos}" y="{margin_top + chart_height + 15}" text-anchor="middle" font-size="9" fill="#7f8c8d">{x_val:.1f}</text>'
            )

    # Draw histogram bars for each condition (slightly offset for visibility)
    conditions = list(data.keys())
    n_conditions = len(conditions)
    sub_bar_width = chart_width / n_bins / (n_conditions + 0.5)

    for cond_idx, cond in enumerate(conditions):
        bins = condition_bins[cond]
        color = COLORS[cond_idx % len(COLORS)]

        for bin_idx, count in enumerate(bins):
            if count == 0:
                continue

            x_left = margin_left + chart_width * bin_idx / n_bins + sub_bar_width * cond_idx
            bar_height = (count / max_count) * chart_height
            y_top = margin_top + chart_height - bar_height

            svg_parts.append(
                f'<rect x="{x_left}" y="{y_top}" width="{sub_bar_width * 0.9}" height="{bar_height}" fill="{color}" opacity="0.7"/>'
            )

    # Legend
    legend_x = width - margin_right + 10
    for i, cond in enumerate(conditions):
        legend_y = margin_top + 20 + i * 20
        label = cond if len(cond) <= 12 else cond[:9] + "..."
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 8}" width="12" height="12" fill="{COLORS[i % len(COLORS)]}" opacity="0.7"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 16}" y="{legend_y}" font-size="10" fill="#2c3e50">{_escape_xml(label)}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def create_means_comparison_svg(
    data: Dict[str, Tuple[float, float]],
    title: str = "Mean Comparison",
    xlabel: str = "Mean Score",
    grand_mean: Optional[float] = None,
    width: int = 600,
    height: int = 300
) -> str:
    """
    Create an SVG dot plot showing means with error bars (horizontal layout).

    Args:
        data: Dict mapping condition names to (mean, std_error) tuples
        title: Chart title
        xlabel: X-axis label
        grand_mean: Optional overall mean to show as reference line
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    if not data:
        return _create_no_data_svg(width, height, "No data available for means comparison")

    margin_left = 120
    margin_right = 30
    margin_top = 50
    margin_bottom = 50
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    conditions = list(data.keys())
    means = [data[c][0] for c in conditions]
    errors = [data[c][1] for c in conditions]
    n_conditions = len(conditions)

    if n_conditions == 0:
        return _create_no_data_svg(width, height, "No conditions to display")

    # Calculate scale
    x_min = min(m - e for m, e in zip(means, errors))
    x_max = max(m + e for m, e in zip(means, errors))
    x_range = x_max - x_min if x_max != x_min else 1
    x_min = x_min - x_range * 0.1
    x_max = x_max + x_range * 0.1

    row_height = min(40, chart_height / n_conditions)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">{_escape_xml(title)}</text>',
        f'<text x="{margin_left + chart_width/2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#2c3e50">{_escape_xml(xlabel)}</text>',
    ]

    # X-axis grid
    n_ticks = 5
    for i in range(n_ticks + 1):
        x_val = x_min + (x_max - x_min) * i / n_ticks
        x_pos = margin_left + chart_width * i / n_ticks

        svg_parts.append(
            f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + chart_height}" stroke="#ecf0f1" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x_pos}" y="{margin_top + chart_height + 15}" text-anchor="middle" font-size="10" fill="#7f8c8d">{x_val:.2f}</text>'
        )

    # Grand mean reference line
    if grand_mean is not None and x_min <= grand_mean <= x_max:
        gm_x = margin_left + (grand_mean - x_min) / (x_max - x_min) * chart_width
        svg_parts.append(
            f'<line x1="{gm_x}" y1="{margin_top}" x2="{gm_x}" y2="{margin_top + chart_height}" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>'
        )
        svg_parts.append(
            f'<text x="{gm_x}" y="{margin_top - 5}" text-anchor="middle" font-size="9" fill="#e74c3c">Grand Mean: {grand_mean:.2f}</text>'
        )

    # Draw dots with error bars
    for i, (cond, mean, error) in enumerate(zip(conditions, means, errors)):
        y_center = margin_top + row_height * (i + 0.5)
        color = COLORS[i % len(COLORS)]

        # Convert to x positions
        x_mean = margin_left + (mean - x_min) / (x_max - x_min) * chart_width
        x_low = margin_left + (mean - error - x_min) / (x_max - x_min) * chart_width
        x_high = margin_left + (mean + error - x_min) / (x_max - x_min) * chart_width

        # Condition label
        label = cond if len(cond) <= 15 else cond[:12] + "..."
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y_center + 4}" text-anchor="end" font-size="11" fill="#2c3e50">{_escape_xml(label)}</text>'
        )

        # Error bar line
        svg_parts.append(
            f'<line x1="{x_low}" y1="{y_center}" x2="{x_high}" y2="{y_center}" stroke="{color}" stroke-width="2"/>'
        )

        # Caps
        svg_parts.append(
            f'<line x1="{x_low}" y1="{y_center - 6}" x2="{x_low}" y2="{y_center + 6}" stroke="{color}" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<line x1="{x_high}" y1="{y_center - 6}" x2="{x_high}" y2="{y_center + 6}" stroke="{color}" stroke-width="2"/>'
        )

        # Mean dot
        svg_parts.append(
            f'<circle cx="{x_mean}" cy="{y_center}" r="6" fill="{color}" stroke="white" stroke-width="2"/>'
        )

        # Value label
        svg_parts.append(
            f'<text x="{x_high + 10}" y="{y_center + 4}" font-size="10" fill="#2c3e50">{mean:.2f}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def create_effect_size_svg(
    effect_size: float,
    effect_type: str = "Cohen's d",
    p_value: Optional[float] = None,
    width: int = 500,
    height: int = 150
) -> str:
    """
    Create an SVG visualization of effect size magnitude.

    Args:
        effect_size: The effect size value
        effect_type: Type of effect size (e.g., "Cohen's d", "Eta squared")
        p_value: Optional p-value to display
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    margin = 50
    bar_width = width - 2 * margin
    bar_height = 30
    bar_y = 70

    # Determine magnitude and color
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        magnitude = "Negligible"
        color = "#95a5a6"
        fill_pct = abs_effect / 0.8 * 0.25
    elif abs_effect < 0.5:
        magnitude = "Small"
        color = "#3498db"
        fill_pct = 0.25 + (abs_effect - 0.2) / 0.3 * 0.25
    elif abs_effect < 0.8:
        magnitude = "Medium"
        color = "#f39c12"
        fill_pct = 0.5 + (abs_effect - 0.5) / 0.3 * 0.25
    else:
        magnitude = "Large"
        color = "#2ecc71"
        fill_pct = min(1.0, 0.75 + (abs_effect - 0.8) / 0.4 * 0.25)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">{_escape_xml(effect_type)}: {effect_size:.3f} ({magnitude})</text>',
    ]

    # Background bar
    svg_parts.append(
        f'<rect x="{margin}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="#ecf0f1" rx="4"/>'
    )

    # Filled portion
    filled_width = bar_width * fill_pct
    svg_parts.append(
        f'<rect x="{margin}" y="{bar_y}" width="{filled_width}" height="{bar_height}" fill="{color}" rx="4"/>'
    )

    # Threshold markers
    thresholds = [(0.2, "0.2\nSmall"), (0.5, "0.5\nMedium"), (0.8, "0.8\nLarge")]
    for thresh, label in thresholds:
        x_pos = margin + bar_width * (thresh / 1.0) * 0.9  # Assuming max displayed is ~1.0
        if x_pos < margin + bar_width:
            svg_parts.append(
                f'<line x1="{x_pos}" y1="{bar_y}" x2="{x_pos}" y2="{bar_y + bar_height}" stroke="#7f8c8d" stroke-width="1" stroke-dasharray="3,3"/>'
            )

    # Scale labels
    svg_parts.append(f'<text x="{margin}" y="{bar_y + bar_height + 15}" font-size="9" fill="#7f8c8d">0</text>')
    svg_parts.append(f'<text x="{margin + bar_width}" y="{bar_y + bar_height + 15}" text-anchor="end" font-size="9" fill="#7f8c8d">Large</text>')

    # P-value if provided
    if p_value is not None:
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        sig_color = "#2ecc71" if p_value < 0.05 else "#e74c3c"
        svg_parts.append(
            f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="{sig_color}">p = {p_value:.4f} {sig}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def create_summary_table_svg(
    data: Dict[str, Dict[str, float]],
    title: str = "Descriptive Statistics",
    width: int = 600,
    height: int = 250
) -> str:
    """
    Create an SVG table showing summary statistics.

    Args:
        data: Dict mapping condition names to dict of statistics {n, mean, std, min, max}
        title: Table title
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    if not data:
        return _create_no_data_svg(width, height, "No data available for summary table")

    conditions = list(data.keys())
    n_rows = len(conditions) + 1  # +1 for header
    row_height = min(35, (height - 60) / n_rows)

    # Column widths
    cols = [("Condition", 120), ("N", 50), ("Mean", 70), ("SD", 70), ("Min", 60), ("Max", 60)]
    total_col_width = sum(w for _, w in cols)
    x_start = (width - total_col_width) / 2
    y_start = 50

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">{_escape_xml(title)}</text>',
    ]

    # Header row background
    svg_parts.append(
        f'<rect x="{x_start}" y="{y_start}" width="{total_col_width}" height="{row_height}" fill="#3498db"/>'
    )

    # Header text
    x_pos = x_start
    for col_name, col_width in cols:
        svg_parts.append(
            f'<text x="{x_pos + col_width/2}" y="{y_start + row_height/2 + 4}" text-anchor="middle" font-size="11" font-weight="bold" fill="white">{col_name}</text>'
        )
        x_pos += col_width

    # Data rows
    for row_idx, cond in enumerate(conditions):
        row_y = y_start + row_height * (row_idx + 1)
        bg_color = "#f8f9fa" if row_idx % 2 == 0 else "white"
        stats = data[cond]

        # Row background
        svg_parts.append(
            f'<rect x="{x_start}" y="{row_y}" width="{total_col_width}" height="{row_height}" fill="{bg_color}"/>'
        )

        # Row border
        svg_parts.append(
            f'<line x1="{x_start}" y1="{row_y + row_height}" x2="{x_start + total_col_width}" y2="{row_y + row_height}" stroke="#dee2e6" stroke-width="1"/>'
        )

        # Cell values
        x_pos = x_start
        values = [
            cond[:15] if len(cond) <= 15 else cond[:12] + "...",
            str(int(stats.get('n', 0))),
            f"{stats.get('mean', 0):.2f}",
            f"{stats.get('std', 0):.2f}",
            f"{stats.get('min', 0):.2f}",
            f"{stats.get('max', 0):.2f}"
        ]

        for (_, col_width), value in zip(cols, values):
            svg_parts.append(
                f'<text x="{x_pos + col_width/2}" y="{row_y + row_height/2 + 4}" text-anchor="middle" font-size="10" fill="#2c3e50">{_escape_xml(value)}</text>'
            )
            x_pos += col_width

    # Table border
    table_height = row_height * (n_rows)
    svg_parts.append(
        f'<rect x="{x_start}" y="{y_start}" width="{total_col_width}" height="{table_height}" fill="none" stroke="#dee2e6" stroke-width="2"/>'
    )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def _create_no_data_svg(width: int, height: int, message: str) -> str:
    """Create a placeholder SVG when no data is available."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">
<rect width="{width}" height="{height}" fill="#f8f9fa"/>
<text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="14" fill="#7f8c8d">{_escape_xml(message)}</text>
</svg>'''
