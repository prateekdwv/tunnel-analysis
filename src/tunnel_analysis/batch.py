"""Resumable batches on local or mounted storage; processing stays local.

State and per-image checkpoints live beside the final output directory. The
final audit is published last and acts as the completion record. This runner
requires one active process per run; mounted Drive is not a locking service.
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
import sys
import tempfile
import time
import uuid
import zipfile

from .config import Config
from .io import sha256
from .pipeline import manifest, overlay_name, write_summary


class BatchError(ValueError):
    pass


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_directory(output):
    output = Path(output).resolve()
    return output.parent / '.tunnel-checkpoints' / output.name


def runtime_identity():
    package = Path(__file__).parent
    code_hash = hashlib.sha256(b''.join(
        p.name.encode() + p.read_bytes() for p in sorted(package.glob('*.py')))).hexdigest()
    commit, dirty = None, None
    try:
        root = subprocess.check_output(
            ['git', '-C', str(package), 'rev-parse', '--show-toplevel'],
            stderr=subprocess.DEVNULL, text=True).strip()
        # Installed wheels may sit inside an unrelated repository.
        if Path(root, 'src', 'tunnel_analysis').resolve() == package.resolve():
            commit = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
            dirty = bool(subprocess.check_output(
                ['git', '-C', root, 'status', '--porcelain', '--untracked-files=all',
                 '--', 'src/tunnel_analysis', 'pyproject.toml'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    return {'code': {'sha256': code_hash, 'git_commit': commit, 'git_dirty': dirty},
            'environment': {'python': platform.python_version(), 'dependencies': dict(sorted(versions.items()))}}


def discover(input_dir):
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'))
    if not files:
        raise BatchError('No TIFF files directly inside the input folder')
    return files


def inventory(input_dir, progress):
    result = []
    for path in discover(input_dir):
        progress(f'Checking source checksum: {path.name}')
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise BatchError(f'Source changed while reading: {path.name}; retry with stable inputs')
        result.append({'name': path.name, 'size': after.st_size, 'sha256': digest})
    return result


def copy_verified(source, destination, expected=None):
    """Stream a copy and verify its bytes by rereading the destination."""
    digest = hashlib.sha256()
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(block)
            dst.write(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise BatchError(f'Checksum changed while copying {Path(source).name}; start a new run for changed inputs')
    if sha256(destination) != value:
        raise OSError(f'Copy verification failed: {destination}')
    return value


def save_state(path, state):
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    if json.loads(temporary.read_text()) != state:
        raise OSError('Checkpoint state verification failed')
    os.replace(temporary, path)


def worker(source, destination, config, sensitivity, scratch):
    """Fresh Python process for every image; no processing against Drive."""
    config_path = scratch / 'config.json'
    config_path.write_text(json.dumps(config.to_dict()))
    command = [sys.executable, '-m', 'tunnel_analysis', str(source), '--output', str(destination),
               '--config', str(config_path), '--_worker']
    if not sensitivity:
        command.append('--no-sensitivity')
    env = os.environ.copy()
    env['TMPDIR'] = str(scratch)
    start = time.perf_counter()
    with (scratch / 'worker-error.txt').open('w+b') as errors:
        completed = subprocess.run(command, env=env, stderr=errors, check=False)
        errors.seek(0, 2)
        end = errors.tell()
        errors.seek(max(0, end - 3000))
        error = errors.read().decode('utf-8', errors='replace').strip()
    if completed.returncode < 0:
        raise InterruptedError('Image worker was terminated; resume this run to retry the unfinished image')
    audit_path = destination / 'audit.zip'
    if audit_path.exists():
        with zipfile.ZipFile(audit_path) as audit:
            if audit.testzip() is not None:
                raise OSError('Worker produced an invalid audit archive')
            records = json.loads(audit.read('run.json'))['results']
        if len(records) != 1:
            raise BatchError('Expected one worker result')
        return records[0]
    if completed.returncode == 0:
        raise BatchError('Worker returned success without an audit archive')
    # Even unsupported TIFFs receive a clearly marked permanent failure preview.
    import numpy as np
    from .report import save_overlay
    destination.mkdir(exist_ok=True)
    record = {'input': str(source), 'status': 'failed', 'error': error or 'Image worker failed',
              'quality_flags': ['processing_error'], 'tunnel_area_px': None,
              'config': config.to_dict(), 'overlay': overlay_name(source),
              'runtime_seconds': time.perf_counter() - start, 'peak_memory_mib': None}
    blank = np.zeros((240,900), dtype=np.uint8)
    save_overlay(destination / record['overlay'], blank, blank, blank, blank, record)
    with zipfile.ZipFile(audit_path, 'w') as audit:
        audit.writestr('run.json', json.dumps(manifest([record])))
    return record


def package_checkpoint(work, record, state, index, target):
    artifacts = ['audit.zip', record['overlay']]
    metadata = {'format_version': 1, 'run_id': state['run_id'], 'index': index,
                'input': state['inputs'][index], 'record': canonical(record),
                'artifacts': {name: sha256(work / name) for name in artifacts}}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('checkpoint.json', json.dumps(metadata))
        for name in artifacts:
            archive.write(work / name, name)


def read_checkpoint(path, state, index, expected_hash):
    """Validate before trusting a checkpoint; never extract arbitrary paths."""
    if sha256(path) != expected_hash:
        raise BatchError('Checkpoint checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('checkpoint.json'))
        overlay = overlay_name(state['inputs'][index]['name'])
        if (metadata['format_version'] != 1 or metadata['run_id'] != state['run_id'] or
                metadata['index'] != index or metadata['input'] != state['inputs'][index] or
                set(metadata['artifacts']) != {'audit.zip', overlay} or
                metadata['record']['overlay'] != overlay or
                metadata['record']['input_sha256'] != state['inputs'][index]['sha256'] or
                len(archive.namelist()) != 3 or
                set(archive.namelist()) != {'checkpoint.json', 'audit.zip', overlay}):
            raise BatchError('Checkpoint does not belong to this image/run')
        for name, digest in metadata['artifacts'].items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != digest:
                raise BatchError('Checkpoint artifact checksum mismatch')
    return metadata


def completed_batch(output):
    with zipfile.ZipFile(output / 'audit.zip') as archive:
        data = json.loads(archive.read('run.json'))
        batch = data['batch']
        if batch['format_version'] != 1 or batch['status'] not in ('complete', 'complete_with_failures'):
            raise BatchError('This folder is not a completed resumable batch')
        if archive.testzip() is not None:
            raise BatchError('Final audit archive is corrupt')
        for record in data['results']:
            for entry in record.get('masks', {}).values():
                with archive.open(entry['archive_path']) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                        raise BatchError('Archived mask checksum mismatch')
    expected = set(batch['artifacts']) | {'audit.zip'}
    actual = {p.name for p in output.iterdir()} - {'.batch-owner.json'}
    if actual != expected:
        raise BatchError('Completed result files are missing or unexpected files were added')
    for name, digest in batch['artifacts'].items():
        if Path(name).name != name or sha256(output / name) != digest:
            raise BatchError('Completed result checksum mismatch')
    return data


def validate_resume(state, input_dir, inputs, identity, config, sensitivity, source_root):
    if state.get('format_version') != 1:
        raise BatchError('Unsupported batch format')
    if state['input_directory'] != str(input_dir) or state['inputs'] != inputs:
        raise BatchError('Input folder, file list, or image contents changed; start a new run')
    if state['identity'] != identity:
        raise BatchError('Code, Git commit, Python, or dependency versions changed; restore the original environment or start a new run')
    if config is not None and canonical(config.to_dict()) != state['config']:
        raise BatchError('Configuration changed; start a new run')
    if sensitivity is not None and sensitivity != state['sensitivity']:
        raise BatchError('Sensitivity setting changed; start a new run')
    if source_root is not None and str(source_root) != state['source_root']:
        raise BatchError('Source root changed; resume with the original source root')


def assemble(state, checkpoints, destination, scratch):
    destination.mkdir()
    records = []
    with zipfile.ZipFile(destination / 'audit.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as final:
        for index, item in enumerate(state['inputs']):
            key = str(index)
            local_checkpoint = scratch / 'checkpoint.zip'
            copy_verified(checkpoints / f'{index:06d}.zip', local_checkpoint, state['checkpoints'][key])
            metadata = read_checkpoint(local_checkpoint, state, index, state['checkpoints'][key])
            record = metadata['record']
            records.append(record)
            with zipfile.ZipFile(local_checkpoint) as package:
                with package.open(record['overlay']) as source, (destination / record['overlay']).open('wb') as target:
                    shutil.copyfileobj(source, target)
                with package.open('audit.zip') as source, (scratch / 'image-audit.zip').open('wb') as target:
                    shutil.copyfileobj(source, target)
            with zipfile.ZipFile(scratch / 'image-audit.zip') as audit:
                for entry in record.get('masks', {}).values():
                    with audit.open(entry['archive_path']) as source, final.open(entry['archive_path'], 'w', force_zip64=True) as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            local_checkpoint.unlink()
            (scratch / 'image-audit.zip').unlink()
        write_summary(records, destination / 'results.csv')
        batch = {key: value for key, value in state.items() if key != 'checkpoints'}
        batch.update(status='complete_with_failures' if any(r['status']=='failed' for r in records) else 'complete',
                     completed_utc=utc_now(),
                     artifacts={p.name: sha256(p) for p in destination.iterdir() if p.name != 'audit.zip'})
        data = manifest(records) | {'batch': batch}
        final.writestr('run.json', json.dumps(data, indent=2) + '\n')
    completed_batch(destination)
    return data


def publish(assembled, output, checkpoints, state):
    output.mkdir(exist_ok=True)
    owner = output / '.batch-owner.json'
    if owner.exists():
        if json.loads(owner.read_text()) != {'run_id': state['run_id']}:
            raise BatchError('Output belongs to a different run')
    elif any(output.iterdir()):
        raise BatchError('Refusing to overwrite an existing results folder')
    else:
        owner.write_text(json.dumps({'run_id': state['run_id']}))
    names = {p.name for p in assembled.iterdir()}
    if {p.name for p in output.iterdir()} - names - {owner.name}:
        raise BatchError('Unexpected files in partial output; refusing to overwrite')
    # Audit is the completion record, so publish it after the other artifacts.
    for name in sorted(names - {'audit.zip'}) + ['audit.zip']:
        temporary = checkpoints / 'publish.part'
        digest = copy_verified(assembled / name, temporary)
        os.replace(temporary, output / name)
        if sha256(output / name) != digest:
            raise OSError(f'Published file verification failed: {name}')
    data = completed_batch(output)
    owner.unlink()
    return data


def cleanup_checkpoints(checkpoints, state):
    if checkpoints.exists():
        stored = json.loads((checkpoints / 'state.json').read_text())
        if stored['run_id'] != state['run_id']:
            raise BatchError('Checkpoint ownership mismatch')
        shutil.rmtree(checkpoints)
        try:
            checkpoints.parent.rmdir()
        except OSError:
            pass


def run_batch(input_dir, output, config=None, sensitivity=None, resume=False,
              source_root=None, scratch_dir=None, progress=print):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else None
    if not input_dir.is_dir():
        raise BatchError('Input must be a directory')
    if output == input_dir or output in input_dir.parents:
        raise BatchError('Output must not be the input folder or an ancestor of it')
    if source_root is not None:
        try:
            input_dir.relative_to(source_root)
        except ValueError as exc:
            raise BatchError('Input must lie inside --source-root') from exc
    checkpoints = checkpoint_directory(output)
    if not resume and (output.exists() or checkpoints.exists()):
        raise BatchError('Run already exists; use --resume or choose a new output directory')
    if resume and not (output.exists() or (checkpoints / 'state.json').exists()):
        raise BatchError('No resumable run found at this location')
    identity = runtime_identity()
    inputs = inventory(input_dir, progress)
    if resume and (output / 'audit.zip').exists():
        try:
            data = completed_batch(output)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            if not (checkpoints / 'state.json').exists():
                raise BatchError('Completed results are damaged and no checkpoints remain; preserve this folder and start a new run')
        else:
            validate_resume(data['batch'], input_dir, inputs, identity, config, sensitivity, source_root)
            owner = output / '.batch-owner.json'
            if owner.exists():
                if json.loads(owner.read_text()) != {'run_id': data['batch']['run_id']}:
                    raise BatchError('Output ownership mismatch')
                owner.unlink()
            cleanup_checkpoints(checkpoints, data['batch'])
            progress('Run already complete; verified results are unchanged')
            return data
    if resume:
        state = json.loads((checkpoints / 'state.json').read_text())
        validate_resume(state, input_dir, inputs, identity, config, sensitivity, source_root)
    else:
        config = config or Config()
        state = {'format_version': 1, 'run_id': uuid.uuid4().hex, 'created_utc': utc_now(),
                 'status': 'running', 'input_directory': str(input_dir), 'output_directory': str(output),
                 'source_root': str(source_root or input_dir), 'inputs': inputs,
                 'config': canonical(config.to_dict()), 'sensitivity': True if sensitivity is None else sensitivity,
                 'identity': identity, 'checkpoints': {}}
        checkpoints.mkdir(parents=True, exist_ok=False)
        save_state(checkpoints / 'state.json', state)
    if state['output_directory'] != str(output):
        raise BatchError('Run belongs to a different output directory')
    config = Config(**(state['config'] | {'ridge_sigmas': tuple(state['config']['ridge_sigmas'])}))
    with tempfile.TemporaryDirectory(prefix='tunnel-batch-', dir=scratch_dir) as temporary:
        scratch = Path(temporary)
        for index, item in enumerate(inputs):
            key = str(index)
            checkpoint = checkpoints / f'{index:06d}.zip'
            if key in state['checkpoints']:
                try:
                    read_checkpoint(checkpoint, state, index, state['checkpoints'][key])
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
                    progress(f'Rebuilding missing or damaged checkpoint: {item["name"]}')
                else:
                    progress(f'Skipping verified completed image: {item["name"]}')
                    continue
            if runtime_identity() != identity:
                raise BatchError('Code or environment changed during the run; restore it before resuming')
            progress(f'Processing {index+1}/{len(inputs)}: {item["name"]}')
            with tempfile.TemporaryDirectory(prefix='image-', dir=scratch) as image_temporary:
                image_scratch = Path(image_temporary)
                local_input = image_scratch / item['name']
                copy_verified(input_dir / item['name'], local_input, item['sha256'])
                work = image_scratch / 'worker-results'
                record = worker(local_input, work, config, state['sensitivity'], image_scratch)
                if runtime_identity() != identity:
                    raise BatchError('Code or environment changed during processing; restore it before resuming')
                if record.get('source_code_sha256', identity['code']['sha256']) != identity['code']['sha256']:
                    raise BatchError('Worker code changed during processing')
                if record.get('input_sha256', item['sha256']) != item['sha256']:
                    raise BatchError('Worker input checksum mismatch')
                record.update(input=str(input_dir / item['name']), input_sha256=item['sha256'],
                              source_relative_path=str((input_dir / item['name']).relative_to(state['source_root'])),
                              git_commit=identity['code']['git_commit'])
                record.setdefault('source_code_sha256', identity['code']['sha256'])
                local_package = image_scratch / 'checkpoint.zip'
                package_checkpoint(work, record, state, index, local_package)
                partial = checkpoints / f'{index:06d}.part'
                digest = copy_verified(local_package, partial)
                read_checkpoint(partial, state, index, digest)
                os.replace(partial, checkpoint)
                state['checkpoints'][key] = digest
                save_state(checkpoints / 'state.json', state)
                progress(f'Saved verified checkpoint: {item["name"]}')
        # Detect edits or newly added images before publishing a completed batch.
        if inventory(input_dir, progress) != inputs:
            raise BatchError('Inputs changed during the run; checkpoints preserved, start a new run')
        if runtime_identity() != identity:
            raise BatchError('Code or environment changed during the run; restore it before resuming')
        progress('Assembling and verifying final results')
        assembled = scratch / 'final'
        assemble(state, checkpoints, assembled, scratch)
        data = publish(assembled, output, checkpoints, state)
    cleanup_checkpoints(checkpoints, state)
    progress(f'Results: {output} ({data["batch"]["status"]})')
    return data
