"""Offline launcher checks: no Drive authorization, GitHub, or pip downloads."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

import pytest


NOTEBOOK = Path(__file__).parents[1] / 'notebooks/Google_Drive_Analysis.ipynb'


def notebook_functions():
    data=json.loads(NOTEBOOK.read_text())
    definitions=[]
    for cell in data['cells']:
        if cell['cell_type']=='code':
            module=ast.parse(''.join(cell['source']))
            definitions.extend(node for node in module.body if isinstance(node,ast.FunctionDef))
    namespace={'Path':Path,'re':re,'subprocess':subprocess,'tempfile':tempfile,'json':json,'zipfile':zipfile}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(NOTEBOOK),'exec'),namespace)
    return namespace


def test_notebook_is_clean_python_launcher():
    notebook=json.loads(NOTEBOOK.read_text())
    assert notebook['nbformat']==4
    assert len({c['id'] for c in notebook['cells']})==len(notebook['cells'])
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert cell['execution_count'] is None and cell['outputs']==[]
            compile(''.join(cell['source']),str(NOTEBOOK),'exec')
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'tunnel_analysis", "batch"' in source
    assert 'drive.mount("/content/drive")' in source
    assert '--no-build-isolation' in source
    assert 'identity["environment"]["python"]' in source


def test_branch_tag_and_commit_resolve_to_detached_code(tmp_path):
    repository=tmp_path/'source';repository.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(repository),*args],text=True,stderr=subprocess.DEVNULL).strip()
    git('init','-b','main')
    git('config','user.name','Local test')
    git('config','user.email','test@example.invalid')
    (repository/'file.txt').write_text('first')
    git('add','file.txt');git('-c','commit.gpgsign=false','commit','-m','first')
    first=git('rev-parse','HEAD')
    git('-c','tag.gpgsign=false','tag','v-test')
    checkout=notebook_functions()['checkout_revision']
    pinned,commit=checkout(str(repository),'main',tmp_path)
    assert commit==first
    (repository/'file.txt').write_text('second')
    git('add','file.txt');git('-c','commit.gpgsign=false','commit','-m','second')
    assert subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()==first
    assert (pinned/'file.txt').read_text()=='first'
    for ref in ('v-test',first):
        folder,commit=checkout(str(repository),ref,tmp_path)
        assert commit==first
        assert (folder/'file.txt').read_text()=='first'
    with pytest.raises(ValueError):checkout(str(repository),'--upload-pack=bad',tmp_path)


def test_notebook_reads_saved_identity_from_state_or_final_audit(tmp_path):
    read=notebook_functions()['read_resume_metadata']
    output=tmp_path/'run'
    state=tmp_path/'.tunnel-checkpoints'/'run'/'state.json';state.parent.mkdir(parents=True)
    metadata={'format_version':1,'identity':{'code':{'git_commit':'a'*40}}}
    state.write_text(json.dumps(metadata))
    assert read(output)==metadata
    state.unlink();output.mkdir()
    with zipfile.ZipFile(output/'audit.zip','w') as archive:
        archive.writestr('run.json',json.dumps({'batch':metadata}))
    assert read(output)==metadata
