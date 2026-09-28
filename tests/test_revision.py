import json
from pathlib import Path
import zipfile
import pytest
from tunnel_analysis import batch
from tunnel_analysis.revision import run_revision
from tunnel_analysis.io import sha256
from test_batch import inputs, fake_worker


def test_revision_carries_other_images_and_resumes(inputs, tmp_path, fake_worker):
    calls, _ = fake_worker
    parent = tmp_path/'parent'
    old = batch.run_batch(inputs, parent, sensitivity=False)
    hashes = {p.name: sha256(p) for p in parent.iterdir()}
    sources = {p.name: sha256(p) for p in inputs.glob('*.TIF')}
    calls.clear()
    out = tmp_path/'revision'
    def stop(message):
        if message.startswith('Saved verified checkpoint:'):
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_revision(inputs, parent, out, ['a photo.TIF'], progress=stop)
    assert calls == ['a photo.TIF']
    new = run_revision(inputs, parent, out, ['a photo.TIF'], resume=True)
    assert calls == ['a photo.TIF']
    assert new['results'][1] == old['results'][1]
    assert (out/old['results'][1]['overlay']).read_bytes() == (parent/old['results'][1]['overlay']).read_bytes()
    with zipfile.ZipFile(parent/'audit.zip') as a, zipfile.ZipFile(out/'audit.zip') as b:
        for mask in old['results'][1]['masks'].values():
            assert a.read(mask['archive_path']) == b.read(mask['archive_path'])
    assert hashes == {p.name: sha256(p) for p in parent.iterdir()}
    assert sources == {p.name: sha256(p) for p in inputs.glob('*.TIF')}
    assert not batch.checkpoint_directory(out).exists()
    assert len(list(out.iterdir())) == 5
    assert new['batch']['revision']['image_configs']['a photo.TIF']['inner_exclusion'] == 'opening-buffer'
    run_revision(inputs, parent, out, ['a photo.TIF'], resume=True)
    with pytest.raises(ValueError, match='already exists'):
        run_revision(inputs, parent, out, ['a photo.TIF'])


def test_revision_selection_failures_and_parent_compatibility(inputs, tmp_path, fake_worker, monkeypatch):
    calls, worker = fake_worker
    parent = tmp_path/'parent'
    batch.run_batch(inputs, parent, sensitivity=False)
    calls.clear()
    with pytest.raises(ValueError, match='exactly match'):
        run_revision(inputs, parent, tmp_path/'bad', ['wrong.tif'])
    def failed(*args):
        r = worker(*args)
        r.update(status='failed', error='Central opening is unbounded or implausible')
        return r
    monkeypatch.setattr(batch, 'worker', failed)
    out=tmp_path/'failed'
    with pytest.raises(ValueError, match='Selected image failed'):
        run_revision(inputs, parent, out, ['a photo.TIF'])
    assert not out.exists()
    assert batch.checkpoint_directory(out).exists()
    monkeypatch.setattr(batch, 'worker', worker)
    run_revision(inputs, parent, out, ['a photo.TIF'], resume=True)
    assert calls == ['a photo.TIF', 'a photo.TIF']
    original = batch.completed_batch
    def incompatible(path):
        data = original(path)
        data['measurement_rule_version'] = 'old'
        return data
    monkeypatch.setattr(batch, 'completed_batch', incompatible)
    with pytest.raises(ValueError, match='Incompatible measurement'):
        run_revision(inputs, parent, tmp_path/'old', ['a photo.TIF'])


def test_revision_multiple_and_changed_source(inputs, tmp_path, fake_worker):
    calls,_=fake_worker
    parent=tmp_path/'parent'
    batch.run_batch(inputs,parent,sensitivity=False)
    calls.clear()
    run_revision(inputs,parent,tmp_path/'both',['a photo.TIF','b photo.tiff'])
    assert calls==['a photo.TIF','b photo.tiff']
    (inputs/'a photo.TIF').write_bytes(b'changed')
    with pytest.raises(ValueError,match='missing or changed'):
        run_revision(inputs,parent,tmp_path/'changed',['a photo.TIF'])


@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_revision_rebuilds_checkpoint(inputs, tmp_path, fake_worker, damage):
    calls,_=fake_worker
    parent=tmp_path/'parent'
    batch.run_batch(inputs,parent,sensitivity=False)
    calls.clear()
    out=tmp_path/'out'
    def stop(message):
        if message.startswith('Saved verified checkpoint:'):
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_revision(inputs,parent,out,['a photo.TIF'],progress=stop)
    checkpoint=batch.checkpoint_directory(out)/'000000.zip'
    if damage=='missing':
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        run_revision(inputs,parent,out,['b photo.tiff'],resume=True)
    run_revision(inputs,parent,out,['a photo.TIF'],resume=True)
    assert calls==['a photo.TIF','a photo.TIF']


def test_revision_cli(inputs, tmp_path, fake_worker):
    from tunnel_analysis.cli import main
    parent=tmp_path/'parent'
    batch.run_batch(inputs,parent,sensitivity=False)
    assert main(['revise',str(inputs),'--from-run',str(parent),'--output',str(tmp_path/'new'),
                 '--image','a photo.TIF','--inner-exclusion','opening-buffer'])==0
