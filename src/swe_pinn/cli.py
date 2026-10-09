"""Command-line workflows; all paths are explicit and relative to --root."""

import argparse
from copy import deepcopy
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train, evaluate and plot SWE, heat and Burgers PINNs"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project/data root (default: current directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--equation", choices=("swe", "heat", "burgers"), default="swe")
    training.add_argument("--config", type=Path, help="default: cases/<equation>/config.yaml")
    training.add_argument("--run-id")
    training.add_argument("--seed", type=int, help="override seed for independent repetitions")
    training.add_argument("--device", help="override device, e.g. cpu or cuda:0")
    training.add_argument("--cpu-threads", type=int)
    training.add_argument("--initial-checkpoint", type=Path, help="heat/Burgers warm start with a fresh optimizer")
    training.add_argument("--options", nargs="+", type=Path, help="optional YAML overrides, applied in order")
    resume = commands.add_parser("resume", help="recover a locally generated run at an epoch boundary")
    resume.add_argument("--run-dir", type=Path, required=True)
    study = commands.add_parser("plot-study", help="paper-style panels for new run outputs")
    study.add_argument("--segment-dirs", nargs="+", type=Path, required=True)
    study.add_argument("--labels", nargs="+")
    study.add_argument("--epochs", nargs="+", type=int)
    study.add_argument("--output", type=Path)
    example = commands.add_parser("check-physics", help="opt-in prescribed-field scientific check")
    example.add_argument("--case", choices=("still_water", "manufactured_fields", "linear_standing_wave"),
                         required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--segment", type=int, default=0)
    evaluation.add_argument("--device", help="override archived device for evaluation")
    evaluation.add_argument("--handoff", type=Path)
    for name in ("plot", "animate"):
        command = commands.add_parser(name)
        if name == "plot":
            paths = command.add_mutually_exclusive_group(required=True)
            paths.add_argument("--segment-dir", type=Path, help="one SWE time segment")
            paths.add_argument("--run-dir", type=Path, help="SWE, heat or Burgers run")
        else:
            command.add_argument("--segment-dir", type=Path, required=True)
        if name == "animate":
            command.add_argument(
                "--kind",
                choices=("solution", "training", "residuals"),
                default="solution",
            )
            command.add_argument("--field", choices=("u", "zeta", "h"), default="zeta")
            command.add_argument("--fps", type=int, default=8)
    reference = commands.add_parser("import-reference")
    reference.add_argument("--u", type=Path, required=True)
    reference.add_argument("--zeta", "--h", dest="zeta", type=Path, required=True,
                           help="elevation array (legacy --h also accepted)")
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument(
        "--dt",
        type=float,
        required=True,
        help="sampling interval of INPUT arrays, seconds",
    )
    reference.add_argument(
        "--dx",
        type=float,
        required=True,
        help="sampling spacing of INPUT arrays, metres",
    )
    reference.add_argument("--x-min", type=float, required=True)
    reference.add_argument("--t-start", type=float, default=0.0)
    reference.add_argument(
        "--source", default="", help="solver settings, provenance, and/or DOI"
    )
    fixture = commands.add_parser(
        "make-fixture",
        help="create SYNTHETIC software-test data, not a scientific reference",
    )
    fixture.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/synthetic.nc")
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def resolve(path):
        return path.resolve() if path.is_absolute() else root / path

    if args.command == "train":
        import yaml
        path = resolve(args.config or Path(f"cases/{args.equation}/config.yaml"))
        equation = (yaml.safe_load(path.read_text()) or {}).get("equation", args.equation)
        if equation == "swe":
            from .training import train
            loader = load_config
        else:
            from .benchmarks.training import train, load_config as loader
        config = loader(path, [resolve(path) for path in args.options or []])
        if args.seed is not None:
            config["seed"] = args.seed
        if args.device is not None:
            config["device"] = args.device
        if args.cpu_threads is not None:
            config["cpu_threads"] = args.cpu_threads
        if args.initial_checkpoint is not None:
            if equation == "swe":
                parser.error("--initial-checkpoint currently applies to heat and Burgers")
            config["training"]["initial_checkpoint"] = str(args.initial_checkpoint)
        print(train(config, root, args.run_id))
    elif args.command == "resume":
        import json
        directory = resolve(args.run_dir)
        if json.loads((directory / "config.json").read_text())["equation"] == "swe":
            from .training import train
            loader = load_config
        else:
            from .benchmarks.training import train, load_config as loader
        print(train(loader(directory / "config.yaml"), root, resume_run=directory))
    elif args.command == "check-physics":
        import json
        from .examples import check
        print(json.dumps(check(args.case), indent=2))
    elif args.command == "evaluate":
        from .training import evaluate_checkpoint

        config = deepcopy(load_config(resolve(args.config)))
        if args.device is not None:
            config["device"] = args.device
        if not 0 <= args.segment < config["domain"]["n_time_segments"]:
            parser.error("segment is outside the configured time segments")
        print(
            evaluate_checkpoint(
                config,
                root,
                resolve(args.checkpoint),
                resolve(args.output),
                args.segment,
                resolve(args.handoff) if args.handoff else None,
            )
        )
    elif args.command == "plot":
        from .plotting import plot_segment
        if args.segment_dir:
            print(plot_segment(resolve(args.segment_dir)))
        else:
            directory = resolve(args.run_dir)
            if (directory / "segments").is_dir():
                for segment in sorted((directory / "segments").iterdir()):
                    if segment.is_dir():
                        print(plot_segment(segment))
            else:
                from .benchmarks.training import plot
                print(plot(directory))
    elif args.command == "plot-study":
        from .plotting import plot_study
        print(plot_study([resolve(path) for path in args.segment_dirs], args.labels,
                         resolve(args.output) if args.output else None, args.epochs))
    elif args.command == "animate":
        from .plotting import animate_segment, animate_history

        if args.fps <= 0:
            parser.error("fps must be positive")
        if args.kind == "solution":
            print(
                animate_segment(
                    resolve(args.segment_dir), args.field, args.fps
                )
            )
        else:
            print(
                animate_history(
                    resolve(args.segment_dir), args.field, args.kind, args.fps
                )
            )
    elif args.command == "import-reference":
        from .reference import import_arrays

        print(
            import_arrays(
                resolve(args.u),
                resolve(args.zeta),
                resolve(args.output),
                args.dt,
                args.dx,
                args.x_min,
                args.t_start,
                args.source,
            )
        )
    else:
        from .fixtures import make_fixture

        print(make_fixture(resolve(args.output)))


if __name__ == "__main__":
    main()
