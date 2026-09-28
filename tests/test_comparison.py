import csv

from PIL import Image
import pytest

from tunnel_analysis.comparison import read_measurements, save_comparison


@pytest.mark.parametrize('values', [
    [('control.tif', '100', '12.5', 'provisional'),
     ('failed.tif', '900', '99', 'failed'),
     ('zero.tif', '0', '0', 'provisional')],
    [('bad.tif', '', '', 'failed'), ('invalid.tif', 'nan', 'inf', 'provisional')],
    [],
])
def test_graph_values_failures_and_determinism(tmp_path, values):
    source = tmp_path / 'results.csv'
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'tunnel_area_px', 'tunnel_area_percent', 'status'])
        writer.writerows(values)
    rows = read_measurements(source)
    for row, original in zip(rows, values):
        if original[0] == 'control.tif':
            assert row['values'] == [12.5, 100]
        elif original[0] == 'zero.tif':
            assert row['values'] == [0, 0]
        else:
            assert row['values'] is None
    first, second = tmp_path / 'first.png', tmp_path / 'second.png'
    save_comparison(source, first)
    save_comparison(source, second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.format == 'PNG'
        assert image.width == 1500
        assert image.height >= 210
