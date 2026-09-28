"""Publication-sized coverage figures; plotting imports stay out of image workers."""
import csv
import math
from pathlib import Path
import re
import textwrap


RULE = 'arena-coverage-v1'
BAR_COLOR = '#0072B2'  # Okabe–Ito blue; labels carry identity without relying on hue.


def display_labels(names):
    """Remove file-format/resolution clutter without merging sample identities."""
    labels = [re.sub(r'\s+', ' ', re.sub(r'\b\d+(?:\.\d+)?\s*dpi\b', '',
              Path(name).stem.replace('_', ' '), flags=re.IGNORECASE)).strip() for name in names]
    for i, label in enumerate(labels.copy()):
        if not label or labels.count(label) > 1:
            labels[i] = names[i]
    # A cleaned label might also equal another original filename.
    return labels if len(set(labels)) == len(labels) else names


def read_measurements(csv_path):
    """Only graph the new denominator; never relabel historical percentages."""
    with open(csv_path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    labels = display_labels([row['image'] for row in rows])
    for row, label in zip(rows, labels):
        row['label'] = label
        row['coverage'] = None
        row['plot_status'] = 'Failed' if row['status'] == 'failed' else 'Invalid measurement'
        if row['status'] == 'failed':
            continue
        if row.get('measurement_rule_version') != RULE:
            raise ValueError('Comparison requires arena-coverage-v1 results; rerun historical images with the new denominator')
        try:
            value = float(row['tunnel_area_percent'])
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(value) and 0 <= value <= 100:
            if 'arena_boundary_clipped_review_coverage' in row.get('notes', ''):
                row['plot_status'] = 'Cropped arena — review'
            else:
                row['coverage'] = value
                row['plot_status'] = row['status']
    return rows


def axis_scale(values):
    """Round to readable ticks, always starting at zero and never above 100%."""
    from matplotlib.ticker import MaxNLocator
    maximum = max(values, default=0.)
    target = min(100., maximum*1.18) if maximum > 0 else 1.
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, target)
    upper = min(100., float(ticks[-1]))
    return upper, [float(t) for t in ticks if 0 <= t <= upper]


def value_label(value):
    return f'{value:.2f}%' if value == 0 or value >= .01 else f'{value:.3g}%'


def create_figure(rows):
    """One bar per image; no implied replicates, tests, or uncertainty intervals."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    labels = [textwrap.fill(row['label'], width=30, break_long_words=True,
                           break_on_hyphens=False) for row in rows]
    positions, cursor = [], 0.
    for label in labels:
        height = max(1., .65*len(label.splitlines()))
        positions.append(cursor+height/2)
        cursor += height
    figure = Figure(figsize=(7.1, max(2.6, 1.55 + .55*cursor)), dpi=100, facecolor='white')
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    figure.subplots_adjust(left=.34, right=.96, bottom=.23, top=.82)
    ax.set_title('Visible tunnel coverage', loc='left', fontsize=13, fontweight='bold', pad=20)
    upper, ticks = axis_scale([row['coverage'] for row in rows if row['coverage'] is not None])
    ax.set_xlim(0, upper)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.set_xlabel('Arena area (%)', labelpad=10)
    ax.set_ylim(max(cursor, 1), 0)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis='y', length=0, pad=12, labelsize=10)
    ax.tick_params(axis='x', length=3, color='#9CA3AF', labelcolor='#374151', labelsize=9)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#9CA3AF')
    ax.spines['bottom'].set_linewidth(.7)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5E7EB', linewidth=.6)
    for row, position in zip(rows, positions):
        value = row['coverage']
        if value is None:
            ax.text(.02*upper, position, row['plot_status'], va='center', fontsize=9,
                    color='#6B7280', fontstyle='italic')
            continue
        ax.barh(position, value, height=.5, color=BAR_COLOR, edgecolor='none')
        inside = value > .88*upper
        ax.annotate(value_label(value), (value, position), xytext=(-6 if inside else 6, 0),
                    textcoords='offset points', va='center', ha='right' if inside else 'left',
                    fontsize=9, color='white' if inside else '#1F2937')
    if not rows:
        ax.text(.5, .5, 'No measurements', transform=ax.transAxes, ha='center', color='#6B7280')
    figure.text(.34, .035, 'Individual images · Provisional measurements', fontsize=8, color='#6B7280')
    return figure


def save_comparison(csv_path, output, svg_path=None):
    """Save a 300-dpi PNG and optional editable SVG, returning audit metadata."""
    import matplotlib
    rows = read_measurements(csv_path)
    settings = {'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': '#1F2937',
                'axes.labelcolor': '#374151', 'svg.fonttype': 'none',
                'svg.hashsalt': RULE, 'text.usetex': False, 'text.parse_math': False}
    with matplotlib.rc_context(settings):
        figure = create_figure(rows)
        try:
            figure.savefig(output, dpi=300, format='png', facecolor='white',
                           metadata={'Title': 'Visible tunnel coverage', 'Software': 'tunnel-analysis'})
            if svg_path is not None:
                figure.savefig(svg_path, format='svg', facecolor='white', metadata={'Date': None})
            xlim = list(figure.axes[0].get_xlim())
            inches = list(figure.get_size_inches())
        finally:
            figure.clear()
    return {'measurement_rule_version': RULE, 'matplotlib_version': matplotlib.__version__,
            'dpi': 300, 'size_inches': inches, 'x_limits_percent': xlim,
            'color': BAR_COLOR, 'source': 'results.csv',
            'rows': [{'image': row['image'], 'label': row['label'], 'coverage_percent': row['coverage'],
                      'status': row['plot_status']} for row in rows],
            'caption': 'Visible tunnel coverage for individual images. Coverage is accepted tunnel pixels '
                       'divided by observed pixels inside the fitted arena, multiplied by 100. Rim and '
                       'debris rejection do not reduce this denominator. Bar colours do not encode '
                       'experimental groups. Failed, invalid, or flagged cropped-arena measurements have '
                       'no bar. No biological replicate uncertainty or statistical tests are represented. '
                       'Measurements remain provisional pending independent validation.'}
