"""CLI. Folder mode uses fresh sequential workers for per-image peak RSS."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import zipfile
from .config import Config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'aggregate-reviewed':
        parser = argparse.ArgumentParser(prog='tunnel-analysis aggregate-reviewed')
        parser.add_argument('runs', nargs='+', type=Path)
        parser.add_argument('--output', required=True, type=Path)
        args = parser.parse_args(argv[1:])
        try:
            from .editor import aggregate_reviewed
            print(aggregate_reviewed(args.runs, args.output))
            return 0
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            print(f'Aggregation error: {exc}', file=sys.stderr)
            return 2
    if argv and argv[0] == "revise":
        return revision_command(argv[1:])
    if argv and argv[0] == "batch":
        return batch_command(argv[1:])
    if argv and argv[0] in ("compare","validate-crops","synthetic","write-config"):
        return utility(argv)
    parser = argparse.ArgumentParser(description="Measure visible white tunnel area in one TIFF or a folder of TIFFs.")
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--no-sensitivity",action="store_true",help="Skip the two additional threshold checks")
    parser.add_argument("--debug-dir",type=Path,help="Explicit separate location for diagnostic outputs")
    parser.add_argument("--validation-dir",type=Path,help="Explicit separate location for annotation crops")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = Config.read(args.config)
        if args._worker:
            from .pipeline import run_image
            result = run_image(args.input,args.output,config,not args.no_sensitivity,
                               lambda s:print(f"[{args.input.name}] {s}",flush=True),
                               args.debug_dir,args.validation_dir)
            if result["status"] == "failed":
                print(result["error"],file=sys.stderr)
                return 2
            print(f"{args.input.name}: {result['tunnel_area_px']:,} px; {result['tunnel_area_percent']:.3f}% of region; provisional",flush=True)
            return 0
        if args.input.is_file():
            files = [args.input]
        elif args.input.is_dir():
            files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in (".tif",".tiff"))
            if not files:
                raise ValueError("No TIFF files in input folder")
        else:
            raise ValueError(f"Input does not exist: {args.input}")
        from .pipeline import check_destinations,write_summary,manifest
        check_destinations(args.output,args.debug_dir,args.validation_dir)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        results = []
        failures = 0
        # Workers write only to temporary locations. Publish one completed batch.
        with tempfile.TemporaryDirectory(prefix="tunnel-workers-") as workers:
            staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-",dir=args.output.parent))
            try:
                with zipfile.ZipFile(staging/"audit.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as audit:
                    for index,path in enumerate(files):
                        destination = Path(workers)/str(index)
                        command = [sys.executable,"-m","tunnel_analysis",str(path),"--output",str(destination),"--_worker"]
                        if args.config:
                            command += ["--config",str(args.config)]
                        if args.no_sensitivity:
                            command += ["--no-sensitivity"]
                        for option,root in (("--debug-dir",args.debug_dir),("--validation-dir",args.validation_dir)):
                            if root:
                                command += [option,str(root/path.name)]
                        completed = subprocess.run(command,check=False)
                        failures += completed.returncode != 0
                        if (destination/"audit.zip").exists():
                            with zipfile.ZipFile(destination/"audit.zip") as worker_audit:
                                results.extend(json.loads(worker_audit.read("run.json"))["results"])
                                for entry in worker_audit.infolist():
                                    if entry.filename == "run.json":
                                        continue
                                    # Stream large masks; never materialize ZIP entries in RAM.
                                    with worker_audit.open(entry) as source, audit.open(entry.filename,"w",force_zip64=True) as target:
                                        shutil.copyfileobj(source,target,length=1024*1024)
                            for overlay in destination.glob("*_overlay.png"):
                                shutil.copyfile(overlay,staging/overlay.name)
                        else:
                            results.append({"input":str(path.resolve()),"status":"failed",
                                            "error":"Input could not be processed; see command error", "quality_flags":["processing_error"]})
                    audit.writestr("run.json",json.dumps(manifest(results),indent=2)+"\n")
                write_summary(results,staging/"results.csv")
                staging.rename(args.output)
            except BaseException:
                shutil.rmtree(staging,ignore_errors=True)
                raise
        print(f"Results: {args.output}/results.csv | overlays | audit.zip",flush=True)
        return 2 if failures else 0
    except (ValueError,OSError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 2


def batch_command(argv):
    parser = argparse.ArgumentParser(prog="tunnel-analysis batch",description="Run or resume a verified batch using local scratch storage.")
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--config",type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-sensitivity",dest="sensitivity",action="store_false")
    group.add_argument("--sensitivity",dest="sensitivity",action="store_true")
    parser.set_defaults(sensitivity=None)
    parser.add_argument("--source-root",type=Path,help="Root for audit source-relative paths, e.g. /content/drive")
    parser.add_argument("--scratch-dir",type=Path,help="Existing local temporary-storage directory")
    args = parser.parse_args(argv)
    try:
        from .batch import run_batch
        result = run_batch(args.input,args.output,Config.read(args.config) if args.config else None,
                           args.sensitivity,args.resume,args.source_root,args.scratch_dir)
        return 2 if result['batch']['status'] == 'complete_with_failures' else 0
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:
        print(f"Batch error: {exc}",file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Batch interrupted. Completed checkpoints are preserved; use --resume.",file=sys.stderr)
        return 130


def utility(argv):
    command = argv[0]
    parser = argparse.ArgumentParser(prog=f"tunnel-analysis {command}")
    if command == "compare":
        parser.add_argument("prediction",type=Path)
        parser.add_argument("reference",type=Path)
    elif command == "validate-crops":
        parser.add_argument("folder",type=Path)
        parser.add_argument("--references",type=Path)
    elif command == "synthetic":
        parser.add_argument("--config",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args(argv[1:])
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        if command == "write-config":
            result = Config().to_dict()
        else:
            from .validation import binary_mask,compare_masks,synthetic_validation,validate_crops
            if command == "synthetic":
                result = synthetic_validation(args.output,Config.read(args.config))
                print(json.dumps(result,indent=2))
                return 0 if all(case.get("acceptance_passed",False) for case in result["cases"]) else 2
            if command == "compare":
                result = compare_masks(binary_mask(args.prediction),binary_mask(args.reference))
            else:
                result = validate_crops(args.folder,args.references)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return 0
    except (ValueError,OSError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 2


def revision_command(argv):
    parser = argparse.ArgumentParser(prog="tunnel-analysis revise")
    parser.add_argument("input", type=Path)
    parser.add_argument("--from-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--inner-exclusion", choices=("opening-buffer", "rim-only"), default="opening-buffer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        from .revision import run_revision
        result = run_revision(args.input, args.from_run, args.output, args.image,
                              args.inner_exclusion, args.resume, args.scratch_dir)
        return 2 if result['batch']['status'] == 'complete_with_failures' else 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Revision error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Revision interrupted; checkpoints preserved. Use revise --resume.", file=sys.stderr)
        return 130
