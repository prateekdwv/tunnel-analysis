"""Local, resumable single-image editing. Widgets are an optional frontend.

Only committed drafts are used by exports. Drive writes are immutable objects
followed by a verified commit record; an interrupted upload cannot replace the
last complete revision. One writer per draft is required and checked.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile

import numpy as np
import tifffile

from .config import Config
from .editor_geometry import Boundaries, confirmed_geometry, suggest_boundaries
from .io import InputImage, sha256
from .pipeline import peak_memory_mib, write_summary
from .report import save_overlay
from .segmentation import feature_maps, segment

RULE = 'reviewed-ellipse-coverage-v1'
MASK_NAMES = ('automatic', 'corrected', 'candidates', 'arena', 'permitted')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]


def identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = ('numpy', 'scipy', 'scikit-image', 'tifffile', 'pillow', 'matplotlib',
                'stackview', 'ipympl', 'ipywidgets', 'ipycanvas', 'ipyevents')
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'source_code_sha256': digest.hexdigest(), 'git_commit': commit,
            'versions': versions, 'python': platform.python_version()}


def copy_checked(source, destination):
    """Checksum while copying; independently verify the destination."""
    digest = hashlib.sha256()
    with open(source, 'rb') as src, open(destination, 'xb') as dst:
        for block in iter(lambda: src.read(1024*1024), b''):
            digest.update(block); dst.write(block)
        dst.flush(); os.fsync(dst.fileno())
    expected = digest.hexdigest()
    if sha256(destination) != expected:
        raise OSError(f'Copy verification failed: {destination}')
    return expected


def verified_publish(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        checksum = copy_checked(source, temporary)
        temporary.rename(destination)
        if sha256(destination) != checksum:
            raise OSError(f'Published file verification failed: {destination}')
        return checksum
    finally:
        temporary.unlink(missing_ok=True)


def read_draft(draft):
    """Read the latest commit. A corrupt committed draft is never skipped silently."""
    files = sorted((Path(draft)/'commits').glob('*.json'))
    if not files:
        raise ValueError('No completed draft checkpoint found')
    state = json.loads(files[-1].read_text())
    if state.get('format') != 1:
        raise ValueError('Unsupported editor draft format')
    for info in state['objects'].values():
        name = info['name']
        if Path(name).name != name:
            raise ValueError('Invalid draft object name')
        path = Path(draft)/'objects'/name
        if not path.is_file() or sha256(path) != info['sha256']:
            raise ValueError(f'Missing or corrupt draft object: {name}')
    return state, files[-1].name


def unpack_object(draft, info, target):
    with zipfile.ZipFile(Path(draft)/'objects'/info['name']) as archive:
        with archive.open('mask.tif') as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst, 1024*1024)
    if sha256(target) != info['mask_sha256']:
        raise ValueError('Unpacked draft mask checksum mismatch')


class EditSession:
    def __init__(self, source, draft, scratch=None, config=None, resume=False):
        self.source, self.draft = Path(source).resolve(), Path(draft).resolve()
        self.config = config or Config()
        self.environment = identity()
        self.work = Path(tempfile.mkdtemp(prefix='tunnel-editor-', dir=scratch))
        self.arrays = {}
        self.active_crop = None
        self.confirmed = False
        self.state = None
        self.head = None
        self.image = None
        self.closed = False
        self.notices = []
        try:
            if not resume and self.draft.exists():
                raise FileExistsError('Draft exists; reopen it with resume=True')
            (self.work/'input').mkdir()
            self.local_source = self.work/'input'/self.source.name
            self.source_hash = copy_checked(self.source, self.local_source)
            self.image = InputImage(self.local_source)
            if resume:
                self.state, self.head = read_draft(self.draft)
                if self.state['source_sha256'] != self.source_hash:
                    raise ValueError('Source image checksum changed; create a new draft')
                # JSON represents tuples as lists; compare canonical forms.
                if (self.state['environment'] != self.environment or
                        json.dumps(self.state['config'], sort_keys=True) != json.dumps(self.config.to_dict(), sort_keys=True)):
                    raise ValueError('Code, dependencies or settings changed; reopen using the saved environment or create a new draft')
                self.boundaries = Boundaries.from_dict(self.state['boundaries'])
                self.confirmed = True
                self._restore()
                self.notices.append('Restored verified draft; automatic analysis is already saved.')
            else:
                self.boundaries, notice = suggest_boundaries(self.image, self.config)
                self.notices.append(notice)
        except BaseException:
            self.close()
            raise

    @property
    def ready(self):
        return self.confirmed and bool(self.arrays) and self.state is not None and self.state['boundaries'] == self.boundaries.to_dict()

    def confirm_boundaries(self, boundaries):
        if self.closed:
            raise ValueError('Session is closed; reopen the image')
        if self.active_crop is not None and self.active_crop.dirty:
            raise ValueError('Apply or discard crop edits before changing boundaries')
        # Check region validity without running expensive native filters.
        confirmed_geometry(self.image, boundaries, self.config)
        self.boundaries, self.confirmed = boundaries, True
        if self.active_crop is not None:
            self.active_crop.stale = True
        if not self.ready and self.state:
            self.notices.append('Boundaries changed. Previous checkpoint preserved; run automatic analysis again.')

    def _close_arrays(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def _restore(self):
        self._close_arrays()
        for name in MASK_NAMES:
            info = self.state['objects'][name]
            target = self.work/f'{name}.tif'
            target.unlink(missing_ok=True)
            unpack_object(self.draft, info, target)
            self.arrays[name] = tifffile.memmap(target, mode='r+' if name == 'corrected' else 'r')
            if self.arrays[name].shape != self.image.shape or self.arrays[name].dtype != np.uint8:
                raise ValueError('Draft mask dimensions or dtype differ from the source')
        # Replay bounded crop patches instead of uploading a whole mask per Apply.
        for edit in self.state['edits']:
            x0, y0, x1, y1 = edit['bbox_xyxy']
            unpack_object(self.draft, self.state['objects'][edit['patch']], self.work/'restore-patch.tif')
            patch = tifffile.imread(self.work/'restore-patch.tif')
            h, w = self.image.shape
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or patch.shape != (y1-y0, x1-x0):
                raise ValueError('Invalid checkpoint crop coordinates')
            if np.any(patch > 1) or np.any((patch > 0) & (self.arrays['permitted'][y0:y1, x0:x1] == 0)):
                raise ValueError('Invalid checkpoint crop pixels')
            self.arrays['corrected'][y0:y1, x0:x1] = patch
        self.arrays['corrected'].flush()

    def analyse(self, sensitivity=False, progress=print):
        if not self.confirmed:
            raise ValueError('Confirm boundaries before analysis')
        if self.active_crop is not None and self.active_crop.dirty:
            raise ValueError('Apply or discard crop edits first')
        if self.ready:
            if bool(self.state.get('sensitivity')) != bool(sensitivity):
                raise ValueError('Sensitivity differs from this saved analysis; use a new draft to change it')
            progress('Using verified automatic result and applied edits.')
            return
        started = time.perf_counter()
        geometry = confirmed_geometry(self.image, self.boundaries, self.config)
        with tempfile.TemporaryDirectory(prefix='analysis-', dir=self.work) as folder:
            folder = Path(folder)
            maps, arrays = [], {}
            try:
                corrected, ridges = feature_maps(self.image, geometry, self.config, folder, progress)
                maps.extend((corrected, ridges))
                for name in MASK_NAMES:
                    arrays[name] = tifffile.memmap(folder/f'{name}.tif', shape=self.image.shape,
                                                  dtype=np.uint8, photometric='minisblack')
                progress('Preserving branch connections and rejecting unsupported patches')
                components = segment(corrected, ridges, geometry, self.config, folder,
                                     arrays['automatic'], arrays['candidates'], arrays['permitted'], source=self.image)
                for y in range(0, self.image.shape[0], 256):
                    end = min(y+256, self.image.shape[0])
                    arrays['arena'][y:end] = self.boundaries.masks(y, end, 0, self.image.shape[1])[0]
                    arrays['corrected'][y:end] = arrays['automatic'][y:end]
                sensitivity_result = None
                if sensitivity:
                    progress('Checking automatic threshold sensitivity (-10%, +10%)')
                    trial = np.memmap(folder/'trial.dat', mode='w+', shape=self.image.shape, dtype=np.uint8)
                    maps.append(trial)
                    sensitivity_result = {}
                    for factor in (.9, 1.1):
                        segment(corrected, ridges, geometry, self.config, folder, trial, factor=factor, source=self.image)
                        sensitivity_result[str(factor)] = int(np.count_nonzero(trial))
                for array in arrays.values():
                    array.flush()
                metrics = {'runtime_seconds': time.perf_counter()-started, 'peak_memory_mib': peak_memory_mib(),
                           'memory_measurement': 'kernel process peak RSS; includes earlier notebook work',
                           'components': components, 'sensitivity': sensitivity_result,
                           'effective_geometry': geometry.describe()}
                progress('Saving verified automatic draft to Drive')
                self._checkpoint({n: folder/f'{n}.tif' for n in MASK_NAMES}, [], metrics)
            finally:
                for array in (*maps, *arrays.values()):
                    array._mmap.close()
        self._restore()
        if self.active_crop:
            self.active_crop.stale = True
        progress('Automatic draft saved. Open a crop to review it.')

    def _checkpoint(self, files, edits, metrics=None):
        commits = sorted((self.draft/'commits').glob('*.json'))
        if (commits[-1].name if commits else None) != self.head:
            raise ValueError('This draft was changed by another session; reopen it before editing')
        objects = dict(self.state['objects']) if self.state and metrics is None else {}
        for name, path in files.items():
            mask_hash = sha256(path)
            packed = self.work/'upload.zip'
            with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                # A fixed timestamp makes identical content produce identical objects.
                info = zipfile.ZipInfo('mask.tif', date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(path, 'rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
            packed_hash = sha256(packed)
            filename = packed_hash + '.zip'
            destination = self.draft/'objects'/filename
            if destination.exists():
                if sha256(destination) != packed_hash:
                    raise ValueError('Corrupt saved object; checkpoint not committed')
            else:
                verified_publish(packed, destination)
            objects[name] = {'name': filename, 'sha256': packed_hash, 'mask_sha256': mask_hash}
        new_state = {'format': 1, 'source': str(self.source), 'source_sha256': self.source_hash,
                     'shape': list(self.image.shape), 'config': self.config.to_dict(),
                     'environment': self.environment, 'boundaries': self.boundaries.to_dict(),
                     'objects': objects, 'edits': edits, 'updated_utc': stamp(),
                     **(metrics if metrics is not None else {k: self.state[k] for k in
                         ('runtime_seconds', 'peak_memory_mib', 'memory_measurement', 'components', 'sensitivity', 'effective_geometry')})}
        path = self.work/'commit.json'
        path.write_text(json.dumps(new_state, indent=2)+'\n')
        name = new_state['updated_utc']+'.json'
        verified_publish(path, self.draft/'commits'/name)
        self.state, self.head = new_state, name

    def crop(self, x, y, size=512):
        if not self.ready:
            raise ValueError('Run analysis with the confirmed boundaries first')
        if self.active_crop and self.active_crop.dirty:
            raise ValueError('Apply or discard edits before switching crops')
        if any(type(v) is not int for v in (x, y, size)) or not 1 <= size <= 1024:
            raise ValueError('Use integer crop coordinates and size between 1 and 1024')
        h, w = self.image.shape
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError('Crop origin is outside the image')
        if self.active_crop:
            self.active_crop.stale = True
        self.active_crop = Crop(self, x, y, min(x+size, w), min(y+size, h))
        return self.active_crop

    def save_reviewed(self, results_parent, reviewer=''):
        if not self.ready:
            raise ValueError('Analysis does not match the confirmed boundaries')
        if self.active_crop and self.active_crop.dirty:
            raise ValueError('Apply or discard unsaved strokes before saving')
        # Verify saved draft before creating a public, immutable revision.
        saved, head = read_draft(self.draft)
        if head != self.head:
            raise ValueError('Draft changed in another session; reopen it')
        destination = Path(results_parent)/('reviewed_'+stamp())
        with tempfile.TemporaryDirectory(prefix='export-', dir=self.work) as folder:
            folder = Path(folder)
            area = arena_area = added = removed = 0
            for y in range(0, self.image.shape[0], 256):
                end = min(y+256, self.image.shape[0])
                mask = self.arrays['corrected'][y:end] > 0
                automatic = self.arrays['automatic'][y:end] > 0
                if np.any(mask & ~(self.arrays['permitted'][y:end] > 0)):
                    raise ValueError('Corrected pixels exist in an excluded region')
                area += int(mask.sum()); arena_area += int(np.count_nonzero(self.arrays['arena'][y:end]))
                added += int(np.count_nonzero(mask & ~automatic)); removed += int(np.count_nonzero(automatic & ~mask))
            if not arena_area:
                raise ValueError('Arena denominator is empty')
            arena = self.boundaries.arena
            missing = max(0., 1-arena_area/(np.pi*arena.radius_x*arena.radius_y))
            flags = ['manual_review_not_independent_validation']
            if missing > self.config.arena_missing_fraction_warning:
                flags.append('arena_boundary_clipped_review_coverage')
            result = {'input': str(self.source), 'input_sha256': self.source_hash,
                      'status': 'reviewed_provisional', 'reviewer': reviewer,
                      'measurement_rule_version': RULE, 'coverage_denominator': 'confirmed full arena; exclusions do not subtract from denominator',
                      'tunnel_area_px': area, 'arena_area_px': arena_area, 'analysis_region_area_px': arena_area,
                      'tunnel_area_percent': 100*area/arena_area, 'quality_flags': flags,
                      'arena_estimated_missing_fraction': missing,
                      'manual_added_px': added, 'manual_removed_px': removed, 'overlay': 'overlay.png',
                      'reviewed_utc': stamp(), 'draft_commit': head, 'session': saved, 'masks': {}}
            save_overlay(folder/'overlay.png', self.image.preview(self.config.preview_size),
                         self.arrays['corrected'], self.arrays['candidates'], self.arrays['permitted'],
                         result, arena=self.arrays['arena'])
            with zipfile.ZipFile(folder/'audit.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for name in MASK_NAMES:
                    source = self.work/f'{name}.tif'
                    archive.write(source, f'masks/{name}.tif')
                    result['masks'][name] = {'archive_path': f'masks/{name}.tif', 'sha256': sha256(source)}
                for edit in saved['edits']:
                    unpack_object(self.draft, saved['objects'][edit['patch']], self.work/'export-patch.tif')
                    archive.write(self.work/'export-patch.tif', f"edits/{edit['patch']}.tif")
                archive.writestr('run.json', json.dumps({'format_version': 1, 'measurement_rule_version': RULE, 'results': [result]}, indent=2))
            write_summary([result], folder/'results.csv')
            publish_folder(folder, destination)
        return destination

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.confirmed = False
        if self.active_crop:
            self.active_crop.stale = True
        self._close_arrays()
        if self.image is not None:
            self.image.close()
        shutil.rmtree(self.work, ignore_errors=True)


class Crop:
    def __init__(self, session, x0, y0, x1, y1):
        self.session = session
        self.bbox = (x0, y0, x1, y1)
        self.slices = (slice(y0, y1), slice(x0, x1))
        self.gray = session.image.gray(*self.slices)
        self.allowed = np.asarray(session.arrays['permitted'][self.slices], bool)
        self.original = np.array(session.arrays['corrected'][self.slices], dtype=np.uint8)
        self.labels = self.original.astype(np.uint32)
        self.history = []
        self.before = None
        self.stale = False

    def check(self):
        if self.stale or not self.session.ready:
            raise ValueError('This crop is stale; open a new crop')

    @property
    def dirty(self):
        return not np.array_equal(self.labels, self.original)

    def begin(self):
        self.check()
        if self.before is None:
            self.before = self.labels.astype(np.uint8)

    def end(self):
        if self.before is not None:
            indices = np.flatnonzero(self.labels.ravel() != self.before.ravel()).astype(np.uint32)
            if indices.size:
                self.history.append((indices, self.before.ravel()[indices].copy()))
                self.history = self.history[-20:]
            self.before = None

    def paint(self, x, y, radius=1, value=1):
        from skimage.draw import disk
        # radius 1 at an integer centre is exactly one native pixel.
        self.begin()
        if value not in (0, 1) or not 1 <= radius <= 64:
            raise ValueError('Use binary labels and brush radius 1–64 native pixels')
        rr, cc = disk((y, x), radius, shape=self.labels.shape)
        if value:
            permitted = self.allowed[rr, cc]
            rr, cc = rr[permitted], cc[permitted]
        self.labels[rr, cc] = value

    def undo(self):
        self.check(); self.end()
        if self.history:
            indices, values = self.history.pop()
            self.labels.ravel()[indices] = values

    def discard(self):
        self.check()
        self.labels[:] = self.original
        self.history.clear(); self.before = None

    def apply(self):
        self.check(); self.end()
        if np.any((self.labels != 0) & (self.labels != 1)) or np.any((self.labels > 0) & ~self.allowed):
            raise ValueError('Crop contains invalid or excluded pixels')
        if not self.dirty:
            return
        target = self.session.arrays['corrected']
        target[self.slices] = self.labels
        target.flush()
        change = {'bbox_xyxy': self.bbox, 'time_utc': stamp(),
                  'patch': f'patch_{len(self.session.state["edits"]):06d}',
                  'added': int(np.count_nonzero((self.labels > 0) & (self.original == 0))),
                  'removed': int(np.count_nonzero((self.labels == 0) & (self.original > 0)))}
        try:
            patch = self.session.work/'patch.tif'
            tifffile.imwrite(patch, self.labels.astype(np.uint8), photometric='minisblack')
            self.session._checkpoint({change['patch']: patch},
                                     self.session.state['edits'] + [change])
        except BaseException:
            target[self.slices] = self.original
            target.flush()
            raise
        self.original[:] = self.labels
        self.history.clear()


def publish_folder(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name('.'+destination.name+'.upload-'+uuid.uuid4().hex)
    staging.mkdir()
    try:
        for path in Path(source).iterdir():
            verified_publish(path, staging/path.name)
        staging.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def aggregate_reviewed(runs, output):
    """Aggregate explicitly chosen revisions; never infer a latest revision."""
    from .comparison import save_comparison
    records, seen = [], set()
    for run in runs:
        with zipfile.ZipFile(Path(run)/'audit.zip') as archive:
            data = json.loads(archive.read('run.json'))
            if data.get('measurement_rule_version') != RULE or len(data['results']) != 1:
                raise ValueError('Only individual reviewed ellipse results can be combined')
            record = data['results'][0]
            if record['status'] != 'reviewed_provisional':
                raise ValueError('Result has not been reviewed')
            key = record['input_sha256']
            name = Path(record['input']).name
            if key in seen or name in seen:
                raise ValueError('Select only one revision per image, with unique image filenames')
            seen.update((key, name))
            # Stream verification; use temporary files for exact native recounts.
            with tempfile.TemporaryDirectory(prefix='reviewed-verify-') as scratch:
                arrays = {}
                try:
                    for mask_name in ('corrected', 'arena', 'permitted'):
                        info = record['masks'][mask_name]
                        path = Path(scratch)/f'{mask_name}.tif'
                        with archive.open(info['archive_path']) as src, open(path, 'wb') as dst:
                            shutil.copyfileobj(src, dst, 1024*1024)
                        if sha256(path) != info['sha256']:
                            raise ValueError('Archived mask checksum mismatch')
                        arrays[mask_name] = tifffile.memmap(path, mode='r')
                    shape = tuple(record['session']['shape'])
                    if any(a.shape != shape or a.dtype != np.uint8 for a in arrays.values()):
                        raise ValueError('Invalid archived mask shape or dtype')
                    area = denominator = 0
                    for y in range(0, shape[0], 256):
                        m, a, p = (arrays[n][y:y+256] for n in ('corrected', 'arena', 'permitted'))
                        if np.any(m > 1) or np.any(a > 1) or np.any(p > 1) or np.any((m > 0) & ((p == 0) | (a == 0))):
                            raise ValueError('Invalid archived mask pixels')
                        area += int(np.count_nonzero(m)); denominator += int(np.count_nonzero(a))
                    if (area != record['tunnel_area_px'] or denominator != record['arena_area_px'] or
                            denominator == 0 or abs(100*area/denominator-record['tunnel_area_percent']) > 1e-10):
                        raise ValueError('Archived masks and measurements disagree')
                finally:
                    for array in arrays.values():
                        array._mmap.close()
            records.append(record)
    if not records:
        raise ValueError('Select at least one reviewed result')
    with tempfile.TemporaryDirectory(prefix='reviewed-comparison-') as folder:
        folder = Path(folder)
        write_summary(records, folder/'results.csv')
        save_comparison(folder/'results.csv', folder/'comparison.png', folder/'comparison.svg', rule=RULE)
        (folder/'sources.json').write_text(json.dumps([{'run': str(Path(p).resolve()), 'audit_sha256': sha256(Path(p)/'audit.zip')} for p in runs], indent=2))
        publish_folder(folder, output)
    return Path(output)
