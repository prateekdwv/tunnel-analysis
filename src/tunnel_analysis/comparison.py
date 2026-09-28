"""Small, deterministic comparison PNGs using the existing Pillow dependency."""
import csv
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


def read_measurements(csv_path):
    """Read displayed values directly from the final CSV; failures have no bars."""
    with open(csv_path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        values = []
        for key in ('tunnel_area_percent', 'tunnel_area_px'):
            try:
                value = float(row[key])
            except (ValueError, TypeError, KeyError):
                value = float('nan')
            values.append(value)
        row['values'] = values if row['status'] != 'failed' and all(
            math.isfinite(v) and v >= 0 for v in values) else None
    return rows


def save_comparison(csv_path, output):
    """Two horizontal bar panels with zero baselines and explicit per-image status."""
    rows = read_measurements(csv_path)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
        title_font = ImageFont.truetype('DejaVuSans.ttf', 23)
    except OSError:
        font = title_font = ImageFont.load_default()
    wrapped = [textwrap.wrap(row['image'], width=34) or ['(unnamed)'] for row in rows]
    heights = [max(68, 22 * (len(lines) + 1)) for lines in wrapped]
    image = Image.new('RGB', (1500, 210 + sum(heights)), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), 'Visible tunnel area: comparison by image', fill='#172b46', font=title_font)
    draw.text((24, 54), 'Individual images; no error bars. Biological accuracy remains provisional.', fill='#444444', font=font)
    panels = [(440, 'Coverage of analysis region (%)'), (990, 'Tunnel area (pixels)')]
    scales = [max([row['values'][i] for row in rows if row['values'] is not None] + [0]) or 1
              for i in range(2)]
    for x, label in panels:
        draw.text((x, 94), label, fill='#172b46', font=font)
    y = 132
    for row, lines, height in zip(rows, wrapped, heights):
        draw.multiline_text((24, y), '\n'.join(lines), fill='#172b46', font=font, spacing=4)
        draw.text((24, y + len(lines) * 22), row['status'], fill='#666666', font=font)
        for i, (x, _) in enumerate(panels):
            draw.line((x, y, x, y + 30), fill='#888888')
            if row['values'] is None:
                draw.text((x + 10, y + 4), 'No measurement', fill='#a43a30', font=font)
                continue
            value = row['values'][i]
            width = 340 * value / scales[i]
            if width > 0:
                draw.rectangle((x + 1, y + 3, x + max(1, width), y + 27), fill='#258579')
            label = f'{value:.2f}%' if i == 0 else f'{value:,.0f}'
            draw.text((x + width + 10, y + 4), label, fill='#172b46', font=font)
        y += height
    draw.text((24, y + 12), 'Both bar scales start at zero. Failed or invalid measurements have no bar.', fill='#444444', font=font)
    draw.text((24, y + 38), 'Pixel areas require matching specimen scale and resolution for direct comparison. Source: results.csv', fill='#444444', font=font)
    image.save(output)
