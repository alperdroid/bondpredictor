"""Command line for the inflation forecasts.

    python -m bondpredictor.inflation build        download data, build feature frames
    python -m bondpredictor.inflation backtest     real-time out-of-sample horse race
    python -m bondpredictor.inflation robustness   seeds, hyperparameters, windows, samples
    python -m bondpredictor.inflation report       markdown report from saved results
    python -m bondpredictor.inflation forecast     latest forecasts
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from ..data import Downloader

CODES = ["US", "UK", "JP", "EA"]


def _frames_path(args) -> Path:
    return Path(args.data)


def cmd_build(args):
    from .data import build_data
    from .features import build_frame

    dl = Downloader(cache_dir=args.cache, offline=args.offline, max_age_days=args.max_age)
    data = build_data(tuple(args.countries), dl=dl, end=args.end)
    frames = {c: build_frame(data, c) for c in args.countries}
    out = _frames_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    notes = {c: data.countries[c].notes for c in args.countries}
    with open(out, "wb") as fh:
        pickle.dump({"frames": frames, "notes": notes, "last_month": data.last_month,
                     "oecd_panel": data.oecd_panel}, fh)
    for c, (f, g) in frames.items():
        pi = f["pi"].dropna()
        print(f"{c}: CPI y/y {pi.index.min()}..{pi.index.max()} (latest {pi.iloc[-1]:.2f}%), {len(g)} features")
        for n in notes[c]:
            print(f"  - {n[:150]}")
    print(f"wrote {out}")


def load_frames(args) -> dict:
    with open(_frames_path(args), "rb") as fh:
        return pickle.load(fh)


def _progress(t0, sink: dict | None = None, path: Path | None = None, every: int = 1):
    """Print each finished job; with ``sink``/``path``, checkpoint results every ``every`` jobs."""
    done = [0]

    def show(key, r):
        if isinstance(r, Exception):
            status = "FAILED"
        elif isinstance(r, str):
            status = r
        else:
            status = f"{len(r)} rows" if isinstance(r, pd.DataFrame) else f"{len(r)} economies"
        print(f"[{time.time() - t0:7.0f}s] {key}: {status}", flush=True)
        if sink is None:
            return
        if isinstance(r, pd.DataFrame):
            sink[key] = r
        elif isinstance(r, dict):  # pooled job: one frame per economy
            for c, rc in r.items():
                sink[(c, key[1], key[2])] = rc
        done[0] += 1
        if path is not None and done[0] % every == 0:
            pd.to_pickle(sink, path)
    return show


def _jobs(frames, specs, countries, horizons, kw):
    return [((c, h, s.name), s, kw) for s in specs for c in countries for h in horizons]


def cmd_backtest(args):
    from .backtest import model_zoo, run_jobs
    from .panel import panel_zoo

    store = load_frames(args)
    frames = {c: store["frames"][c] for c in args.countries}
    end = args.end or str(store["last_month"])
    kw = dict(start=args.start, end=end, min_train=args.min_train)
    args.horizons = args.horizons or [3, 6, 12, 24]
    specs = [s for s in model_zoo(args.tier) if not args.models or s.name in args.models]
    jobs = _jobs(frames, specs, args.countries, args.horizons, {**kw, "explain": True})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.pkl"
    results = pd.read_pickle(path) if (args.resume and path.exists()) else {}
    jobs = [j for j in jobs if j[0] not in results]
    t0 = time.time()
    print(f"{len(jobs)} country jobs on {args.jobs} processes", flush=True)
    raw = run_jobs(frames, jobs, args.jobs, progress=_progress(t0, results, path))
    pd.to_pickle(results, path)
    if not args.no_panel:
        panel_specs = [s for s in panel_zoo() if not args.models or s.name in args.models]
        pjobs = [(("panel", h, s.name), s, kw) for s in panel_specs for h in args.horizons]
        pjobs = [j for j in pjobs if not any((c, j[0][1], j[0][2]) in results for c in args.countries)]
        print(f"{len(pjobs)} pooled jobs", flush=True)
        run_panel_jobs(frames, pjobs, args.jobs, progress=_progress(t0, results, path, every=1))
        pd.to_pickle(results, path)
    failed = [k for k, v in raw.items() if isinstance(v, Exception)]
    print(f"wrote {path} ({len(results)} forecast series); failed: {failed or 'none'}")


_PANEL = {}


def _run_panel_job(job):
    import threadpoolctl

    from .panel import run_panel_backtest

    key, spec, kw = job
    with threadpoolctl.threadpool_limits(limits=1):
        try:
            return key, run_panel_backtest(_PANEL, key[1], spec, **kw)
        except Exception as err:
            logging.getLogger(__name__).exception("pooled job %s failed", key)
            return key, err


def run_panel_jobs(frames, jobs, n_jobs, progress=None):
    global _PANEL
    _PANEL = frames
    out = {}
    if n_jobs <= 1:
        for j in jobs:
            k, r = _run_panel_job(j)
            out[k] = r
            if progress:
                progress(k, r)
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
        for f in as_completed([pool.submit(_run_panel_job, j) for j in jobs]):
            k, r = f.result()
            out[k] = r
            if progress:
                progress(k, r)
    return out


def cmd_robustness(args):
    from .backtest import run_jobs
    from .robustness import robustness_specs

    store = load_frames(args)
    frames = {c: store["frames"][c] for c in args.countries}
    end = args.end or str(store["last_month"])
    kw = dict(start=args.start, end=end, min_train=args.min_train)
    out_dir = Path(args.out)
    path = out_dir / "robustness.pkl"
    results = pd.read_pickle(path) if (args.resume and path.exists()) else {}
    horizons = args.horizons or [12]
    specs = [s for s in robustness_specs() if not args.only or any(o in s.name for o in args.only)]
    jobs = [((c, h, s.name), s, kw) for s in specs for c in args.countries for h in horizons]
    jobs = [j for j in jobs if j[0] not in results]
    t0 = time.time()
    print(f"{len(jobs)} robustness jobs on {args.jobs} processes", flush=True)
    run_jobs(frames, jobs, args.jobs, progress=_progress(t0, results, path))
    pd.to_pickle(results, path)
    print(f"wrote {path}")


def cmd_deep(args):
    import torch

    from .deep import run_deep

    torch.set_num_threads(args.threads)
    store = load_frames(args)
    t0 = time.time()
    res = run_deep(store, args.countries, names=tuple(args.models), horizons=tuple(args.horizons or [3, 6, 12, 24]),
                   start=args.start, end=args.end, refit_every=args.refit_every, max_steps=args.max_steps,
                   progress=_progress(t0))
    path = Path(args.out) / "deep.pkl"
    pd.to_pickle(res, path)
    print(f"wrote {path} ({len(res)} forecast series)")


def cmd_foundation(args):
    import torch

    from .foundation import run_foundation

    torch.set_num_threads(args.threads)
    store = load_frames(args)
    t0 = time.time()
    res = run_foundation(store, args.countries, names=tuple(args.models), horizons=tuple(args.horizons or [3, 6, 12, 24]),
                         start=args.start, end=args.end, progress=_progress(t0))
    path = Path(args.out) / "foundation.pkl"
    pd.to_pickle(res, path)
    print(f"wrote {path} ({len(res)} forecast series)")


def cmd_explain(args):
    from .explain import drivers_report

    store = load_frames(args)
    text = drivers_report({c: store["frames"][c] for c in args.countries})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}")


def cmd_report(args):
    from .report import build_report

    store = load_frames(args)
    results = pd.read_pickle(Path(args.results) / "results.pkl")
    rob_path = Path(args.results) / "robustness.pkl"
    robust = pd.read_pickle(rob_path) if rob_path.exists() else {}
    for extra in ("deep.pkl", "foundation.pkl"):
        p = Path(args.results) / extra
        if p.exists():
            results = {**results, **pd.read_pickle(p)}
    fc_path = Path(args.results) / "latest.csv"
    latest = pd.read_csv(fc_path) if fc_path.exists() else None
    text, tables = build_report(results, robust, store, args.countries, args.horizons, latest, n_boot=args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    tables.to_csv(out.with_suffix(".csv"), index=False)
    print(text)
    print(f"\nwrote {out} and {out.with_suffix('.csv')}")


def cmd_forecast(args):
    from .forecast import latest_forecasts

    store = load_frames(args)
    frames = {c: store["frames"][c] for c in args.countries}
    table = latest_forecasts(frames, args.horizons, n_jobs=args.jobs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "latest.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {out / 'latest.csv'}")


def build_parser():
    p = argparse.ArgumentParser(prog="bondpredictor.inflation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--countries", nargs="+", default=CODES, choices=CODES)
    common.add_argument("--data", default="data/processed/inflation.pkl")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--horizons", nargs="+", type=int, default=None, help="default 3 6 12 24 (robustness: 12)")
    run.add_argument("--start", default="2000-01", help="first forecast origin")
    run.add_argument("--end", default=None, help="last forecast origin (default: last complete month)")
    run.add_argument("--min-train", type=int, default=120)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--resume", action="store_true", help="skip jobs already in the output file")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", parents=[common])
    b.add_argument("--cache", default=None)
    b.add_argument("--offline", action="store_true")
    b.add_argument("--max-age", type=float, default=1.0)
    b.add_argument("--end", default=None, help="last complete month (default: previous calendar month)")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("backtest", parents=[common, run])
    t.add_argument("--tier", type=int, default=1, help="1: benchmarks + econometric ML")
    t.add_argument("--models", nargs="+", default=None)
    t.add_argument("--no-panel", action="store_true")
    t.add_argument("--out", default="output/inflation")
    t.set_defaults(func=cmd_backtest)

    # Parent parsers share their actions, so the robustness default (h = 12) is applied in cmd_robustness.
    r = sub.add_parser("robustness", parents=[common, run])
    r.add_argument("--only", nargs="+", default=None, help="run variants whose name contains any of these strings")
    r.add_argument("--out", default="output/inflation")
    r.set_defaults(func=cmd_robustness)

    dp = sub.add_parser("deep", parents=[common, run], help="tier 2: global deep time-series networks")
    dp.add_argument("--models", nargs="+", default=["nhits", "patchtst", "tide", "lstm"])
    dp.add_argument("--refit-every", type=int, default=24)
    dp.add_argument("--max-steps", type=int, default=600)
    dp.add_argument("--threads", type=int, default=4)
    dp.add_argument("--out", default="output/inflation")
    dp.set_defaults(func=cmd_deep)

    fm = sub.add_parser("foundation", parents=[common, run], help="tier 2: pretrained foundation models, zero-shot")
    fm.add_argument("--models", nargs="+", default=["chronos2", "chronos2_cov", "chronos_bolt", "timesfm25", "timesfm3",
                                                    "timesfm3_cov"])
    fm.add_argument("--threads", type=int, default=4)
    fm.add_argument("--out", default="output/inflation")
    fm.set_defaults(func=cmd_foundation)

    ex = sub.add_parser("explain", parents=[common], help="predictor significance and importance")
    ex.add_argument("--out", default="docs/inflation_drivers.md")
    ex.set_defaults(func=cmd_explain)

    rep = sub.add_parser("report", parents=[common])
    rep.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    rep.add_argument("--results", default="output/inflation")
    rep.add_argument("--n-boot", type=int, default=2000)
    rep.add_argument("--out", default="docs/inflation_report.md")
    rep.set_defaults(func=cmd_report)

    f = sub.add_parser("forecast", parents=[common])
    f.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--out", default="output/inflation")
    f.set_defaults(func=cmd_forecast)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
