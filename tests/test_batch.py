from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
import pytest
import tifffile

from tunnel_analysis import batch
from tunnel_analysis.config import Config
from tunnel_analysis.cli import main
from tunnel_analysis.io import sha256
from tunnel_analysis.validation import synthetic_scene, audit_mask


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / 'photos with spaces'
    folder.mkdir()
    for name in ('a photo.TIF', 'b photo.tiff'):
        tifffile.imwrite(folder / name, np.ones((32,32), np.uint8))
    (folder / 'ignored.jpg').write_bytes(b'not a tiff')
    (folder / 'nested').mkdir()
    tifffile.imwrite(folder / 'nested' / 'not scanned.tif', np.ones((32,32),np.uint8))
    return folder


@pytest.fixture
def fake_worker(monkeypatch):
    calls = []
    def run(source, destination, config, sensitivity, scratch):
        calls.append(source.name)
        assert scratch in source.parents
        destination.mkdir()
        overlay = source.name + '_overlay.png'
        Image.new('RGB',(32,32),'green').save(destination / overlay)
        record = {'input':str(source),'input_sha256':sha256(source),
                  'status':'provisional','tunnel_area_px':64,'analysis_region_area_px':1024,
                  'tunnel_area_percent':6.25,'arena_area_px':1024,
                  'measurement_rule_version':'arena-coverage-v1','quality_flags':[], 'overlay':overlay, 'masks':{}}
        with zipfile.ZipFile(destination/'audit.zip','w') as archive:
            for name,mask in [('tunnels',np.pad(np.ones((8,8),np.uint8),12)),
                              ('analysis_region',np.ones((32,32),np.uint8))]:
                stream=io.BytesIO();tifffile.imwrite(stream,mask)
                data=stream.getvalue();entry=f'images/{source.name}/{name}.tif'
                archive.writestr(entry,data)
                record['masks'][name]={'archive_path':entry,'sha256':hashlib.sha256(data).hexdigest()}
            archive.writestr('run.json',json.dumps({'results':[record]}))
        return record
    monkeypatch.setattr(batch,'worker',run)
    return calls,run


def interrupt_after_first(inputs, output):
    def progress(message):
        if message.startswith('Saved verified checkpoint:'):
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(inputs,output,sensitivity=False,progress=progress)
    return batch.checkpoint_directory(output)


def test_resumed_and_uninterrupted_results_and_sources_match(inputs,tmp_path,fake_worker):
    calls,_=fake_worker
    before={p.name:sha256(p) for p in inputs.iterdir() if p.is_file()}
    out=tmp_path/'resumed'
    checkpoints=interrupt_after_first(inputs,out)
    assert not out.exists()
    assert (checkpoints/'000000.zip').exists()
    data=batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert calls==['a photo.TIF','b photo.tiff']
    assert not checkpoints.exists()
    assert data['batch']['status']=='complete'
    assert [r['source_relative_path'] for r in data['results']]==['a photo.TIF','b photo.tiff']
    assert len(list(out.iterdir()))==5
    direct=tmp_path/'direct'
    batch.run_batch(inputs,direct,sensitivity=False,progress=lambda _:None)
    assert (out/'results.csv').read_bytes()==(direct/'results.csv').read_bytes()
    assert (out/'comparison.png').read_bytes()==(direct/'comparison.png').read_bytes()
    assert data['batch']['artifacts']['comparison.png']==sha256(out/'comparison.png')
    with zipfile.ZipFile(out/'audit.zip') as archive:
        assert 'figures/comparison.svg' in archive.namelist()
        metadata=json.loads(archive.read('figures/comparison.json'))
        assert metadata['rows'][0]['coverage_percent']==6.25
        assert hashlib.sha256(archive.read('figures/comparison.svg')).hexdigest()==data['batch']['figure']['sha256']
    for index in range(2):
        np.testing.assert_array_equal(audit_mask(out,image_index=index),audit_mask(direct,image_index=index))
    assert before=={p.name:sha256(p) for p in inputs.iterdir() if p.is_file()}
    before_calls=len(calls)
    hashes={p.name:sha256(p) for p in out.iterdir()}
    batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert len(calls)==before_calls
    assert hashes=={p.name:sha256(p) for p in out.iterdir()}
    with pytest.raises(batch.BatchError,match='already exists'):
        batch.run_batch(inputs,out)


@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_bad_checkpoint_is_recomputed(inputs,tmp_path,fake_worker,damage):
    calls,_=fake_worker
    out=tmp_path/'result'
    cp=interrupt_after_first(inputs,out)/'000000.zip'
    if damage=='missing':cp.unlink()
    else:cp.write_bytes(b'broken')
    batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert calls==['a photo.TIF','a photo.TIF','b photo.tiff']
    assert batch.completed_batch(out)['batch']['status']=='complete'


@pytest.mark.parametrize('change',['contents','added','removed','config','sensitivity','code','dependencies','python'])
def test_resume_rejects_changed_inputs_or_settings(inputs,tmp_path,fake_worker,monkeypatch,change):
    calls,_=fake_worker
    out=tmp_path/'result';interrupt_after_first(inputs,out)
    options={}
    if change=='contents':
        path=inputs/'a photo.TIF';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    elif change=='added':shutil.copyfile(inputs/'a photo.TIF',inputs/'c.tif')
    elif change=='removed':(inputs/'b photo.tiff').unlink()
    elif change=='config':options['config']=replace(Config(),ridge_threshold=.2)
    elif change=='sensitivity':options['sensitivity']=True
    else:
        identity=batch.runtime_identity()
        if change=='code':identity['code']['sha256']='different'
        elif change=='python':identity['environment']['python']='different'
        else:identity['environment']['dependencies']['numpy']='different'
        monkeypatch.setattr(batch,'runtime_identity',lambda:identity)
    with pytest.raises(batch.BatchError,match='changed'):
        batch.run_batch(inputs,out,resume=True,progress=lambda _:None,**options)
    assert calls==['a photo.TIF']


def test_partial_publication_resumes_without_reanalysis(inputs,tmp_path,fake_worker,monkeypatch):
    calls,_=fake_worker
    out=tmp_path/'result'
    original=batch.copy_verified
    def interrupted(source,destination,expected=None):
        if Path(destination).name=='publish.part':raise KeyboardInterrupt
        return original(source,destination,expected)
    monkeypatch.setattr(batch,'copy_verified',interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(inputs,out,sensitivity=False,progress=lambda _:None)
    assert (out/'.batch-owner.json').exists()
    monkeypatch.setattr(batch,'copy_verified',original)
    batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert len(calls)==2
    assert len(list(out.iterdir()))==5
    assert not batch.checkpoint_directory(out).exists()


def test_completion_before_cleanup_is_recoverable(inputs,tmp_path,fake_worker,monkeypatch):
    calls,_=fake_worker
    original=batch.cleanup_checkpoints
    monkeypatch.setattr(batch,'cleanup_checkpoints',lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    out=tmp_path/'result'
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(inputs,out,sensitivity=False,progress=lambda _:None)
    assert (out/'audit.zip').exists()
    monkeypatch.setattr(batch,'cleanup_checkpoints',original)
    batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert len(calls)==2
    assert not batch.checkpoint_directory(out).exists()


def test_damaged_completed_run_is_preserved(inputs,tmp_path,fake_worker):
    out=tmp_path/'result';batch.run_batch(inputs,out,sensitivity=False,progress=lambda _:None)
    (out/'results.csv').write_text('modified')
    with pytest.raises(batch.BatchError,match='damaged'):
        batch.run_batch(inputs,out,resume=True,progress=lambda _:None)
    assert (out/'results.csv').read_text()=='modified'


def test_real_worker_and_failure_overlay_continue(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    tifffile.imwrite(inputs/'a bad.tif',np.zeros((3,64,64),np.uint8),photometric='minisblack')
    image,_,_=synthetic_scene(256)
    tifffile.imwrite(inputs/'b good.tif',image)
    source_hashes=[sha256(p) for p in sorted(inputs.iterdir())]
    out=tmp_path/'out'
    assert main(['batch',str(inputs),'--output',str(out),'--no-sensitivity','--source-root',str(tmp_path)])==2
    data=batch.completed_batch(out)
    assert data['batch']['status']=='complete_with_failures'
    assert [r['status'] for r in data['results']]==['failed','provisional']
    assert data['results'][1]['source_relative_path']=='input/b good.tif'
    assert np.count_nonzero(audit_mask(out,image_index=1))==data['results'][1]['tunnel_area_px']
    assert all((out/(p.name+'_overlay.png')).exists() for p in inputs.iterdir())
    assert len(list(out.iterdir()))==5
    assert source_hashes==[sha256(p) for p in sorted(inputs.iterdir())]
    assert main(['batch',str(inputs),'--output',str(out),'--resume'])==2
    direct=tmp_path/'direct'
    from tunnel_analysis.pipeline import run_image
    run_image(inputs/'b good.tif',direct,Config(),False)
    np.testing.assert_array_equal(audit_mask(out,image_index=1),audit_mask(direct))


def test_cli_invalid_destinations(inputs,tmp_path):
    assert main(['batch',str(inputs),'--output',str(inputs)])==2
    assert main(['batch',str(inputs),'--output',str(tmp_path/'absent'),'--resume'])==2


def test_real_resumed_and_uninterrupted_masks_and_overlays_match(tmp_path):
    inputs=tmp_path/'real input';inputs.mkdir()
    for index,noisy in enumerate((False,True)):
        image,_,_=synthetic_scene(192,noisy=noisy)
        tifffile.imwrite(inputs/f'{index} scene.tif',image)
    interrupted=tmp_path/'interrupted'
    interrupt_after_first(inputs,interrupted)
    resumed=batch.run_batch(inputs,interrupted,resume=True,progress=lambda _:None)
    direct=tmp_path/'direct'
    normal=batch.run_batch(inputs,direct,sensitivity=False,progress=lambda _:None)
    assert (interrupted/'results.csv').read_bytes()==(direct/'results.csv').read_bytes()
    for index,(a,b) in enumerate(zip(resumed['results'],normal['results'])):
        assert a['tunnel_area_px']==b['tunnel_area_px']
        np.testing.assert_array_equal(audit_mask(interrupted,image_index=index),audit_mask(direct,image_index=index))
        np.testing.assert_array_equal(audit_mask(interrupted,'analysis_region',index),audit_mask(direct,'analysis_region',index))
        assert (interrupted/a['overlay']).read_bytes()==(direct/b['overlay']).read_bytes()
