import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import tifffile

from tunnel_analysis.config import Config
from tunnel_analysis.editor import EditSession, aggregate_reviewed, read_draft
from tunnel_analysis.editor_geometry import Boundaries, Ellipse, confirmed_geometry
from tunnel_analysis.regions import tile_fields
from tunnel_analysis.io import sha256
from tunnel_analysis.validation import synthetic_scene


@pytest.fixture
def session(tmp_path):
    source = tmp_path/'source with spaces.tif'
    image, _, _ = synthetic_scene(180)
    tifffile.imwrite(source, image, photometric='minisblack')
    s = EditSession(source, tmp_path/'draft', config=Config(tile_size=96, preview_size=180))
    s.confirm_boundaries(Boundaries(Ellipse(90,90,86,86), Ellipse(90,90,80,78,.2), Ellipse(88,92,20,15,-.3)))
    s.analyse(progress=lambda _:None)
    yield s
    s.close()


def test_scaled_rotated_masks_and_native_tiles(tmp_path):
    for scale in (1,2,3):
        b = Boundaries(Ellipse(40*scale,35*scale,30*scale,25*scale,1.2),
                       Ellipse(39*scale,36*scale,27*scale,21*scale,1.2),
                       Ellipse(41*scale,34*scale,7*scale,5*scale,-1.1))
        a,p = b.masks(0,80*scale,0,90*scale)
        if scale == 1:
            original_a,original_p = a,p
        else:
            np.testing.assert_array_equal(a[::scale,::scale],original_a)
            np.testing.assert_array_equal(p[::scale,::scale],original_p)
        tiled = np.zeros_like(p)
        for y in range(0,len(p),17):
            for x in range(0,p.shape[1],19):
                tiled[y:y+17,x:x+19] = b.masks(y,min(y+17,len(p)),x,min(x+19,p.shape[1]))[1]
        np.testing.assert_array_equal(tiled,p)
        assert not np.any(p & ~a)


def test_manual_geometry_matches_tile_fields(session):
    g = confirmed_geometry(session.image,session.boundaries,session.config)
    _,expected = session.boundaries.masks(0,180,0,180)
    actual,_ = tile_fields(g,(180,180),0,180,0,180,session.config)
    np.testing.assert_array_equal(actual,expected)
    assert not np.any(session.arrays['automatic'][~expected])


def test_single_pixel_and_edge_edit_undo_exclusion(session):
    c = session.crop(20,70,40)
    ys,xs = np.nonzero(c.allowed & (c.labels==0))
    y,x = int(ys[0]),int(xs[0])
    c.paint(x,y,1);c.end()
    assert np.count_nonzero(c.labels != c.original) == 1
    c.undo(); assert not c.dirty
    for x,y in ((0,0),(39,0),(0,39),(39,39)):
        c.paint(x,y,5)
    c.end()
    assert not np.any(c.labels[~c.allowed])
    c.discard();assert not c.dirty
    with pytest.raises(ValueError):session.crop(0,0,1025)


def test_apply_resume_overlap_sources_and_export(session, tmp_path, monkeypatch):
    from tunnel_analysis import editor
    from tunnel_analysis.cli import main
    original_overlay = editor.save_overlay
    displayed = {}
    def capture_overlay(*args, **kwargs):
        displayed['accepted'] = original_overlay(*args, **kwargs)
        return displayed['accepted']
    monkeypatch.setattr(editor, 'save_overlay', capture_overlay)
    before = sha256(session.source)
    c = session.crop(25,65,50)
    c.paint(20,20,4,1);c.end();c.apply()
    expected = session.arrays['corrected'].copy()
    c2 = session.crop(35,75,50)
    np.testing.assert_array_equal(c2.labels[:40,:40],expected[75:115,35:75])
    c2.paint(10,10,1,0);c2.end();c2.apply()
    expected = session.arrays['corrected'].copy()
    state,_ = read_draft(session.draft)
    assert len(state['edits']) == 2
    assert (session.draft/'objects'/state['objects']['patch_000001']['name']).stat().st_size < 5000
    reopened = EditSession(session.source,session.draft,config=session.config,resume=True)
    try:
        np.testing.assert_array_equal(expected,reopened.arrays['corrected'])
        old_head = reopened.head
        reopened.analyse(progress=lambda _:None)
        assert reopened.head == old_head
        output = reopened.save_reviewed(tmp_path/'results','test')
        assert set(p.name for p in output.iterdir()) == {'results.csv','overlay.png','audit.zip'}
        np.testing.assert_array_equal(displayed['accepted'], expected > 0)
        with zipfile.ZipFile(output/'audit.zip') as archive:
            record = json.loads(archive.read('run.json'))['results'][0]
            with archive.open('masks/corrected.tif') as stream:
                mask = tifffile.imread(stream)
            np.testing.assert_array_equal(mask,expected)
            assert record['tunnel_area_px'] == np.count_nonzero(expected)
            assert record['arena_area_px'] == np.count_nonzero(session.arrays['arena'])
            assert record['tunnel_area_percent'] == 100*record['tunnel_area_px']/record['arena_area_px']
        assert main(['aggregate-reviewed', str(output), '--output', str(tmp_path/'comparison')]) == 0
        assert (tmp_path/'comparison/comparison.png').exists()
        with pytest.raises(FileExistsError):aggregate_reviewed([output], tmp_path/'comparison')
        with pytest.raises(ValueError,match='one revision'):aggregate_reviewed([output,output],tmp_path/'bad')
    finally:
        reopened.close()
    assert sha256(session.source) == before


def test_interrupted_apply_preserves_old_draft(session, monkeypatch):
    from tunnel_analysis import editor
    c = session.crop(25,65,50)
    c.paint(20,20,4,1);c.end()
    old,head = read_draft(session.draft)
    publish = editor.verified_publish
    def fail_commit(src,dst):
        if Path(dst).suffix == '.json':raise OSError('interrupted')
        return publish(src,dst)
    monkeypatch.setattr(editor,'verified_publish',fail_commit)
    with pytest.raises(OSError,match='interrupted'):c.apply()
    assert read_draft(session.draft)[1] == head
    np.testing.assert_array_equal(session.arrays['corrected'][c.slices],c.original)
    assert c.dirty
    monkeypatch.setattr(editor,'verified_publish',publish)
    c.apply();assert not c.dirty


def test_draft_corruption_and_changed_source_settings(session, tmp_path):
    with pytest.raises(ValueError,match='settings'):
        EditSession(session.source,session.draft,config=Config(tile_size=64),resume=True)
    with pytest.raises(FileExistsError):EditSession(session.source,session.draft)
    changed = tmp_path/'changed.tif'
    tifffile.imwrite(changed,np.zeros((180,180),np.uint8))
    with pytest.raises(ValueError,match='checksum'):
        EditSession(changed,session.draft,config=session.config,resume=True)
    state,_ = read_draft(session.draft)
    (session.draft/'objects'/state['objects']['automatic']['name']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        EditSession(session.source,session.draft,config=session.config,resume=True)


def test_boundary_changes_and_unsaved_edits_invalidate(session, tmp_path):
    c = session.crop(25,65,50)
    c.paint(20,20,3);c.end()
    with pytest.raises(ValueError,match='Apply or discard'):session.crop(0,0,20)
    with pytest.raises(ValueError,match='Apply or discard'):session.save_reviewed(tmp_path/'results')
    c.apply()
    _,previous = read_draft(session.draft)
    session.confirm_boundaries(Boundaries(session.boundaries.arena,Ellipse(90,90,70,70)))
    assert not session.ready
    with pytest.raises(ValueError,match='match'):session.save_reviewed(tmp_path/'results')
    session.analyse(progress=lambda _:None)
    assert session.ready
    assert (session.draft/'commits'/previous).exists()
    assert session.state['edits'] == []
    with pytest.raises(ValueError,match='stale'):c.paint(1,1)


def test_code_change_rejected_on_resume(session, monkeypatch):
    from tunnel_analysis import editor
    current = editor.identity()
    monkeypatch.setattr(editor,'identity',lambda:{**current,'source_code_sha256':'changed'})
    with pytest.raises(ValueError,match='Code'):
        EditSession(session.source,session.draft,config=session.config,resume=True)


def test_stackview_pointer_events_match_native_pixels(session):
    pytest.importorskip('stackview')
    from tunnel_analysis.notebook_editor import StackviewBrush
    c = session.crop(25,65,50)
    brush = StackviewBrush(c,zoom=2)
    try:
        brush.size.value = 1
        event = brush.events[0]
        c.labels[:]=0;c.original[:]=0
        def send(kind,x,y,buttons=1):
            event.handle(dict(type=kind,relativeX=x*2,relativeY=y*2,buttons=buttons,altKey=False))
        send('mousedown',20,20);send('mouseup',20,20,0)
        assert brush.error is None
        assert c.labels[20,20] == 1 and np.count_nonzero(c.labels) == 1
        send('mousedown',30,20);send('mousemove',34,20);send('mouseup',34,20,0)
        assert not np.any(c.labels[20,21:30]) # no bridge between separate strokes
        assert np.all(c.labels[20,30:35])
        c.undo();assert np.count_nonzero(c.labels) == 1
        brush.mode.value='Erase'
        send('mousedown',20,20);send('mouseup',20,20,0)
        assert not c.labels.any()
        brush.size.value=10
        for x,y in ((0,0),(49,0),(0,49),(49,49)):
            send('mousedown',x,y);send('mouseleave',x,y,0)
        assert brush.error is None
        brush.visible.value=False
        assert brush.alpha_cell.cell_contents == 0
    finally:
        brush.close()


def test_boundary_widgets_and_pixel_display(session):
    pytest.importorskip('ipympl')
    import matplotlib.pyplot as plt
    previous = plt.get_backend()
    plt.switch_backend('module://ipympl.backend_nbagg')
    from tunnel_analysis.notebook_editor import BoundaryEditor, MaskEditor
    try:
        b = BoundaryEditor(session)
        b.kind.value = 'inner'
        b.enabled.value = True
        b.fields['center_x'].value = 88
        b.fields['radius_x'].value = 25
        b.fields['radius_y'].value = 17
        b.fields['angle'].value = 72
        assert not session.confirmed
        b._detail(None)
        assert b.detail_artist.get_array().shape == (180,180)
        b._confirm(None)
        e = session.boundaries.inner
        assert e.radius_x == 17 and e.radius_y == 25
        assert np.degrees(e.angle) == pytest.approx(-18)
        session.analyse(progress=lambda _:None)
        m = MaskEditor(session,session.work/'results')
        m.x.value=25;m.y.value=65;m.size.value=256
        m.open_crop()
        sample = np.arange(12,dtype=np.uint8).reshape(3,4)
        m.brush.view.zoom_factor=4
        np.testing.assert_array_equal(m.brush.view._zoom(sample),np.repeat(np.repeat(sample,4,axis=0),4,axis=1))
        m.brush.close()
        plt.close(b.fig);plt.close(m.fig)
    finally:
        plt.switch_backend(previous)


def test_deterministic_analysis_and_tiles(tmp_path):
    image,_,_ = synthetic_scene(180)
    source=tmp_path/'scene.tif';tifffile.imwrite(source,image)
    masks=[]
    for i,tile in enumerate((96,96,512)):
        s=EditSession(source,tmp_path/f'draft{i}',config=Config(tile_size=tile,preview_size=180))
        try:
            s.confirm_boundaries(Boundaries(Ellipse(90,90,86,86),Ellipse(90,90,80,78,.2),Ellipse(88,92,20,15,-.3)))
            s.analyse(progress=lambda _:None)
            masks.append(s.arrays['automatic'].copy())
        finally:s.close()
    for m in masks[1:]:np.testing.assert_array_equal(m,masks[0])


def test_manual_boundaries_when_detection_fails_and_missing_checkpoint(tmp_path):
    source=tmp_path/'dark.tif';tifffile.imwrite(source,np.zeros((100,100),np.uint8))
    s=EditSession(source,tmp_path/'draft')
    try:
        assert 'failed' in s.notices[0]
        s.confirm_boundaries(Boundaries(Ellipse(50,50,49,49),Ellipse(50,50,45,45)))
        s.analyse(progress=lambda _:None)
        assert np.count_nonzero(s.arrays['automatic']) == 0
        state,_=read_draft(s.draft)
        (s.draft/'objects'/state['objects']['permitted']['name']).unlink()
        with pytest.raises(ValueError,match='Missing'):read_draft(s.draft)
    finally:s.close()


def test_incompatible_aggregation_and_no_partial_publish(tmp_path,session):
    bad=tmp_path/'old';bad.mkdir()
    with zipfile.ZipFile(bad/'audit.zip','w') as z:z.writestr('run.json',json.dumps({'measurement_rule_version':'arena-coverage-v1','results':[]}))
    with pytest.raises(ValueError,match='Only individual'):aggregate_reviewed([bad],tmp_path/'combined')
    assert not (tmp_path/'combined').exists()


def test_editor_notebook_is_clean_and_one_image_launcher():
    path=Path(__file__).parents[1]/'notebooks/Google_Drive_Mask_Editor.ipynb'
    notebook=json.loads(path.read_text())
    sources=[]
    for c in notebook['cells']:
        if c['cell_type']=='code':
            assert c['outputs']==[] and c['execution_count'] is None
            text=''.join(c['source']);compile(text,str(path),'exec');sources.append(text)
    text='\n'.join(sources)
    assert 'output.enable_custom_widget_manager()' in text
    assert 'RUN_SENSITIVITY = False' in text
    assert 'EditSession(source,draft' in text
    assert 'brush_demo(session.image)' in text
    assert 'MaskEditor(session' in text


def test_interrupted_reanalysis_preserves_manual_checkpoint(session,monkeypatch):
    from tunnel_analysis import editor
    c=session.crop(25,65,50);c.paint(20,20,4);c.end();c.apply()
    old_state,old_head=read_draft(session.draft)
    old_mask=session.arrays['corrected'].copy()
    session.confirm_boundaries(Boundaries(session.boundaries.arena,Ellipse(90,90,70,70)))
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(editor,'segment',interrupted)
    with pytest.raises(KeyboardInterrupt):session.analyse(progress=lambda _:None)
    assert read_draft(session.draft)[1]==old_head
    np.testing.assert_array_equal(session.arrays['corrected'],old_mask)
    assert old_state['edits']


def test_corrupt_measurement_export_is_rejected(session,tmp_path):
    output=session.save_reviewed(tmp_path/'results')
    with zipfile.ZipFile(output/'audit.zip') as z:
        contents={n:z.read(n) for n in z.namelist()}
    data=json.loads(contents['run.json'])
    data['results'][0]['tunnel_area_px']+=1
    contents['run.json']=json.dumps(data).encode()
    with zipfile.ZipFile(output/'audit.zip','w') as z:
        for n,payload in contents.items():z.writestr(n,payload)
    with pytest.raises(ValueError,match='disagree'):aggregate_reviewed([output],tmp_path/'badcomparison')
    assert not (tmp_path/'badcomparison').exists()


def test_widget_close_and_display_size_limit(session):
    pytest.importorskip('stackview')
    from tunnel_analysis.notebook_editor import StackviewBrush
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='smaller crop'):
        StackviewBrush(SimpleNamespace(labels=np.zeros((513,513),np.uint32)),zoom=2)
    crop=session.crop(0,0,180)
    brush=StackviewBrush(crop)
    view=brush.view
    brush.close();brush.close()
    assert view.comm is None
    session.close()
    with pytest.raises(ValueError,match='stale'):crop.paint(1,1)


def test_input_filename_cannot_collide_with_working_mask(tmp_path):
    source=tmp_path/'corrected.tif'
    pixels=np.zeros((100,100),np.uint8);pixels[48:52,10:90]=255
    tifffile.imwrite(source,pixels)
    s=EditSession(source,tmp_path/'draft')
    try:
        s.confirm_boundaries(Boundaries(Ellipse(50,50,49,49),Ellipse(50,50,45,45)))
        s.analyse(progress=lambda _:None)
        np.testing.assert_array_equal(s.image.gray(),pixels.astype(np.float32)/255)
        assert s.local_source.exists()
        np.testing.assert_array_equal(tifffile.imread(source),pixels)
    finally:s.close()
