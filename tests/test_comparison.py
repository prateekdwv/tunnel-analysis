import csv
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from tunnel_analysis.comparison import (read_measurements, save_comparison, create_figure,
                                        axis_scale, display_labels)


def write_csv(path, values, rule='arena-coverage-v1'):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'tunnel_area_percent', 'status', 'measurement_rule_version', 'notes'])
        for row in values:
            writer.writerow([*row[:3], rule, row[3] if len(row) > 3 else ''])


@pytest.mark.parametrize('values', [
    [('control.tif', '12.5', 'provisional'), ('failed.tif', '99', 'failed'),
     ('zero.tif', '0', 'provisional')],
    [('bad.tif', '', 'failed'), ('invalid.tif', 'nan', 'provisional')],
    [],
])
def test_graph_values_failures_and_determinism(tmp_path, values):
    source = tmp_path/'results.csv'
    write_csv(source, values)
    rows = read_measurements(source)
    for row, original in zip(rows, values):
        expected = {'control.tif': 12.5, 'zero.tif': 0}.get(original[0])
        assert row['coverage'] == expected
    first, second, svg = tmp_path/'first.png', tmp_path/'second.png', tmp_path/'figure.svg'
    metadata = save_comparison(source, first, svg)
    vector = svg.read_bytes()
    save_comparison(source, second, svg)
    assert first.read_bytes() == second.read_bytes()
    assert vector == svg.read_bytes()
    assert ET.fromstring(vector).tag.endswith('svg')
    assert b'<text' in vector  # editable text, not a raster embedded in SVG
    with Image.open(first) as image:
        assert image.format == 'PNG'
        assert image.width >= 2100
        assert image.info['dpi'][0] == pytest.approx(300, abs=.01)
    assert metadata['dpi'] == 300
    assert len(metadata['rows']) == len(values)


def test_single_panel_lengths_zero_axis_and_failure_not_zero(tmp_path):
    source = tmp_path/'results.csv'
    write_csv(source, [('Control 300 dpi.tif', '15.19', 'provisional'),
                       ('NSP3.tif', '6.04', 'provisional'),
                       ('failed.tif', '', 'failed'), ('zero.tif', '0', 'provisional')])
    figure = create_figure(read_measurements(source))
    try:
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.get_xlim()[0] == 0
        assert [p.get_width() for p in ax.patches] == [15.19, 6.04, 0]
        assert [t.get_text() for t in ax.get_yticklabels()][0] == 'Control'
        assert 'Failed' in [t.get_text() for t in ax.texts]
    finally:
        figure.clear()


@pytest.mark.parametrize('values', [[0], [], [.00003], [.63, 15.19], [99, 100]])
def test_scale_has_readable_zero_based_ticks(values):
    upper, ticks = axis_scale(values)
    assert 0 < upper <= 100
    assert upper >= max(values, default=0)
    assert ticks[0] == 0
    assert 2 <= len(ticks) <= 7


def test_legacy_percentages_are_not_relabelled(tmp_path):
    source = tmp_path/'results.csv'
    write_csv(source, [('old.tif', '15.19', 'provisional')], rule='central-tunnels-v1')
    with pytest.raises(ValueError, match='rerun historical'):
        read_measurements(source)


def test_out_of_range_and_cropped_measurements_not_plotted(tmp_path):
    source = tmp_path/'results.csv'
    write_csv(source, [('negative.tif', '-1', 'provisional'), ('over.tif', '101', 'provisional'),
                       ('crop.tif', '12', 'provisional', 'arena_boundary_clipped_review_coverage')])
    assert all(row['coverage'] is None for row in read_measurements(source))


def test_labels_preserve_identity_and_literal_characters(tmp_path):
    names = ['sample 300 dpi.tif', 'sample 600 dpi.tif', 'sample.tiff']
    assert len(set(display_labels(names))) == len(names)
    source = tmp_path/'results.csv'
    write_csv(source, [('Very long sample_name with $literal$ experimental details and a replicate 001.tif', '100', 'provisional')])
    svg = tmp_path/'figure.svg'
    save_comparison(source, tmp_path/'figure.png', svg)
    assert '$literal$' in svg.read_text()
