"""Selected-image revisions; parent results are immutable."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

from . import batch
from .config import Config
from .io import sha256
from .pipeline import manifest


def run_revision(input_dir, parent, output, images, inner_exclusion='opening-buffer',
                 resume=False, scratch_dir=None, progress=print):
    input_dir, parent, output = (Path(p).resolve() for p in (input_dir, parent, output))
    if output == parent or output in parent.parents or parent in output.parents:
        raise batch.BatchError('Revision output must be separate from the parent run')
    if output == input_dir or output in input_dir.parents:
        raise batch.BatchError('Output must not replace the input folder')
    if not images or len(set(images)) != len(images):
        raise batch.BatchError('Select one or more unique exact image filenames')
    data = batch.completed_batch(parent)
    records = data['results']
    names = [Path(r['input']).name for r in records]
    if len(set(names)) != len(names):
        raise batch.BatchError('Parent contains ambiguous duplicate filenames')
    if any(Path(n).name != n or n not in names for n in images):
        raise batch.BatchError('Selected images must exactly match filenames in the parent run')
    if (data.get('measurement_rule_version') != 'arena-coverage-v1' or
            any(r.get('measurement_rule_version') != 'arena-coverage-v1'
                for r in records if r['status'] != 'failed')):
        raise batch.BatchError('Incompatible measurement definition: revision requires full-arena coverage. '
                               'Create a new full batch for an older analysis-region run.')
    selected = sorted(images)
    # Only selected sources are read. Unselected sources need not be mounted.
    inputs = []
    configs = {}
    for r in records:
        name = Path(r['input']).name
        item = dict(next(i for i in data['batch']['inputs'] if i['name'] == name))
        if name in selected:
            path = input_dir / name
            progress(f'Checking selected source checksum: {name}')
            if not path.is_file() or sha256(path) != item['sha256']:
                raise batch.BatchError(f'Selected source missing or changed: {name}')
            cfg = r.get('config', data['batch']['config'])
            configs[name] = batch.canonical(replace(Config(**cfg), inner_exclusion=inner_exclusion).to_dict())
        inputs.append(item)
    identity = batch.runtime_identity()
    spec = {'parent_directory': str(parent), 'parent_audit_sha256': sha256(parent/'audit.zip'),
            'selected_images': selected, 'inner_exclusion': inner_exclusion,
            'image_configs': configs, 'parent_identity': data['batch']['identity']}
    checkpoints = batch.checkpoint_directory(output)
    if not resume and (output.exists() or checkpoints.exists()):
        raise batch.BatchError('Run already exists; use --resume or choose a new output directory')
    if resume:
        if (output/'audit.zip').exists():
            done = batch.completed_batch(output)
            state = done['batch']
        else:
            state = json.loads((checkpoints/'state.json').read_text())
        if (state.get('revision') != spec or state['identity'] != identity or
                state['input_directory'] != str(input_dir) or state['output_directory'] != str(output)):
            raise batch.BatchError('Revision inputs, parent, selection, settings, or environment changed')
        if (output/'audit.zip').exists():
            owner = output/'.batch-owner.json'
            if owner.exists():
                if json.loads(owner.read_text()) != {'run_id': state['run_id']}:
                    raise batch.BatchError('Output ownership mismatch')
                owner.unlink()
            batch.cleanup_checkpoints(checkpoints, state)
            return done
    else:
        state = {'format_version': 1, 'run_id': uuid.uuid4().hex, 'created_utc': batch.utc_now(),
                 'status': 'running', 'input_directory': str(input_dir), 'output_directory': str(output),
                 'source_root': str(input_dir), 'inputs': inputs, 'config': data['batch']['config'],
                 'sensitivity': data['batch']['sensitivity'], 'identity': identity,
                 'revision': spec, 'checkpoints': {}}
        checkpoints.mkdir(parents=True)
        batch.save_state(checkpoints/'state.json', state)
    with tempfile.TemporaryDirectory(prefix='tunnel-revision-', dir=scratch_dir) as temp:
        scratch = Path(temp)
        for index, (item, original) in enumerate(zip(inputs, records)):
            name, key = item['name'], str(index)
            if key in state['checkpoints']:
                try:
                    saved = batch.read_checkpoint(checkpoints/f'{index:06d}.zip', state, index, state['checkpoints'][key])
                    if saved['record']['status'] == 'failed' and name in selected:
                        raise batch.BatchError('Retry selected failed image')
                except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                    progress(f'Rebuilding checkpoint: {name}')
                else:
                    progress(f'Skipping verified checkpoint: {name}')
                    continue
            with tempfile.TemporaryDirectory(dir=scratch) as image_temp:
                local = Path(image_temp)
                work = local/'results'
                if name in selected:
                    progress(f'Processing selected image: {name}')
                    source = local/name
                    batch.copy_verified(input_dir/name, source, item['sha256'])
                    record = batch.worker(source, work, Config(**configs[name]), state['sensitivity'], local)
                    if batch.runtime_identity() != identity or record.get('source_code_sha256', identity['code']['sha256']) != identity['code']['sha256']:
                        raise batch.BatchError('Code or environment changed during processing')
                    if record.get('input_sha256', item['sha256']) != item['sha256']:
                        raise batch.BatchError('Worker input checksum mismatch')
                    record.update(input=str(input_dir/name), input_sha256=item['sha256'],
                                  source_relative_path=name, git_commit=identity['code']['git_commit'],
                                  source_code_sha256=identity['code']['sha256'], revision_identity=identity)
                else:
                    progress(f'Carrying forward unchanged result: {name}')
                    record = original
                    work.mkdir()
                    batch.copy_verified(parent/record['overlay'], work/record['overlay'])
                    with zipfile.ZipFile(parent/'audit.zip') as src, zipfile.ZipFile(work/'audit.zip', 'w') as dst:
                        for entry in record.get('masks', {}).values():
                            with src.open(entry['archive_path']) as a, dst.open(entry['archive_path'], 'w', force_zip64=True) as b:
                                shutil.copyfileobj(a, b, 1024*1024)
                        dst.writestr('run.json', json.dumps(manifest([record])))
                package = local/'checkpoint.zip'
                batch.package_checkpoint(work, record, state, index, package)
                partial = checkpoints/f'{index:06d}.part'
                digest = batch.copy_verified(package, partial)
                batch.read_checkpoint(partial, state, index, digest)
                os.replace(partial, checkpoints/f'{index:06d}.zip')
                state['checkpoints'][key] = digest
                batch.save_state(checkpoints/'state.json', state)
                progress(f'Saved verified checkpoint: {name}')
                if name in selected and record['status'] == 'failed':
                    raise batch.BatchError(f'Selected image failed: {name}: {record.get("error", "processing failed")}. '
                                           'Checkpoints preserved; original results unchanged.')
        if sha256(parent/'audit.zip') != spec['parent_audit_sha256']:
            raise batch.BatchError('Parent run changed during revision')
        batch.completed_batch(parent)
        for item in inputs:
            if item['name'] in selected and sha256(input_dir/item['name']) != item['sha256']:
                raise batch.BatchError('Selected source changed during revision')
        if batch.runtime_identity() != identity:
            raise batch.BatchError('Code or environment changed during revision')
        batch.assemble(state, checkpoints, scratch/'final', scratch)
        result = batch.publish(scratch/'final', output, checkpoints, state)
    batch.cleanup_checkpoints(checkpoints, state)
    progress(f'Revised results: {output}')
    return result
