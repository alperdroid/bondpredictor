"""Command line for the inflation forecasts.

    python -m bondpredictor.inflation build        download data, build feature frames
    python -m bondpredictor.inflation backtest     real-time out-of-sample horse race
    python -m bondpredictor.inflation robustness   seeds, hyperparameters, windows, samples
    python -m bondpredictor.inflation report       markdown report from saved results
    python -m bondpredictor.inflation forecast     latest forecasts
    python -m bondpredictor.inflation scenarios    what-if responses of the latest forecasts
    python -m bondpredictor.inflation site         interactive site (docs/site/index.html)
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
    jobs = [] if args.panel_only else [j for j in jobs if j[0] not in results]
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
    from .report import build_report, latest_section, replace_latest

    if args.latest_only:  # new data: refresh the latest-forecast tables, keep the backtest sections
        res_dir, out = Path(args.results), Path(args.out)
        fm = pd.read_pickle(res_dir / "foundation.pkl") if (res_dir / "foundation.pkl").exists() else {}
        lines = latest_section(pd.read_csv(res_dir / "latest.csv"), fm, args.countries)
        out.write_text(replace_latest(out.read_text(), lines))
        print("\n".join(lines))
        print(f"\nupdated the latest forecasts in {out}")
        return
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


def cmd_scenarios(args):
    from .scenario import run_scenarios, summarise

    store = load_frames(args)
    frames = {c: store["frames"][c] for c in args.countries}
    path = Path(args.out) / "scenarios.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_pickle(path) if args.resume and path.exists() else {}
    t0 = time.time()

    def save(key, r):
        done[key] = r
        print(f"[{time.time() - t0:7.0f}s] {key}: {'ok' if r else 'no fit'}", flush=True)
        pd.to_pickle(done, path)

    res = run_scenarios(frames, tuple(args.horizons), n_jobs=args.jobs, progress=save, done=done)
    pd.to_pickle(res, path)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summarise(res).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {path} ({len(res)} model fits)")


def cmd_site(args):
    import json

    from .site import previous_data, site_data, write_site

    store = load_frames(args)
    res_dir = Path(args.results)
    results = {}
    for name in ("results.pkl", "deep.pkl", "foundation.pkl"):
        if (res_dir / name).exists():
            results = {**results, **pd.read_pickle(res_dir / name)}
    # Without a backtest on disk, keep the backtest parts of the page built last time.
    prev_path = Path(args.previous or args.out)
    previous = None
    if not (res_dir / "results.pkl").exists():
        previous = previous_data(prev_path.read_text())
        print(f"no {res_dir / 'results.pkl'}: reusing the backtest from {prev_path}")
    scen_path = res_dir / "scenarios.pkl"
    data = site_data(store, results, pd.read_csv(res_dir / "latest.csv"),
                     pd.read_pickle(scen_path) if scen_path.exists() else None, pd.read_csv(args.scores),
                     {c: store["frames"][c] for c in CODES}, previous=previous)
    (res_dir / "site.json").write_text(json.dumps(data))
    out = write_site(data, Path(args.template), Path(args.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def cmd_transforms(args):
    from ..data import Downloader as DL
    from .data import build_data
    from .transform_search import SCREEN_VERSION, compare_table, run_compare, run_screen

    store = load_frames(args)
    data = build_data(tuple(CODES), dl=DL(offline=True), end=str(store["last_month"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    screen_path = out / "transform_screen.pkl"
    stale = screen_path.exists() and pd.read_pickle(screen_path).get("version") != SCREEN_VERSION
    if args.stage == "screen" or (args.stage == "compare" and (not screen_path.exists() or stale)):
        res = run_screen(data, store, tuple(CODES), tuple(args.horizons), progress=_progress(t0))
        pd.to_pickle(res, screen_path)
        print(f"wrote {screen_path} ({len(res['table'])} candidates)")
    if args.stage == "report":
        from .transform_search import report

        tables = [out / f"transform_compare_{c}_table.csv" for c in CODES]
        compare = pd.concat([pd.read_csv(t) for t in tables if t.exists()]) if any(t.exists() for t in tables) else None
        latest_path = out / "transform_latest_US.csv"
        # Follow-up rounds: the estimated core anchor (round 3) and the US underlying measures (round 2).
        parts = [pd.read_csv(f) for f in sorted((out / "round3").glob("transform_compare_*_table.csv"))]
        r2 = out / "round2" / "transform_compare_US_table.csv"
        if r2.exists():
            parts.append(pd.read_csv(r2).query("set == 'underlying'"))
        rounds = pd.concat(parts) if parts else None
        r4 = out / "round4" / "transform_compare_US_UK_JP_EA_table.csv"
        text = report(pd.read_pickle(screen_path), compare, store["frames"],
                      pd.read_csv(latest_path) if latest_path.exists() else None, rounds,
                      pd.read_csv(r4) if r4.exists() else None)
        Path(args.report).write_text(text)
        print(text)
        print(f"wrote {args.report}")
        return
    if args.stage != "compare":
        return
    table = pd.read_pickle(screen_path)["table"]
    tag = "_".join(args.countries)
    extras = None
    if {"underlying", "underlying_anchor"} & set(args.sets):
        from .transform_search import underlying_features

        dl = DL(cache_dir=args.cache, max_age_days=args.max_age)
        extras = {c: underlying_features(dl, c, store["frames"][c][0].index) for c in args.countries}
    res = run_compare(data, store, table, args.countries, tuple(args.horizons), tuple(args.sets), tuple(args.models),
                      n_jobs=args.jobs, progress=_progress(t0), extras=extras)
    pd.to_pickle(res, out / f"transform_compare_{tag}.pkl")
    long = pd.concat([r.assign(economy=c, horizon=h, set=s, model=m).reset_index()[
        ["economy", "horizon", "set", "model", "date", "forecast", "rw", "actual"]]
        for (c, h, s, m), r in res.items() if isinstance(r, pd.DataFrame) and not r.empty])
    long["date"] = long["date"].astype(str)
    long.to_csv(out / f"transform_compare_{tag}.csv.gz", index=False)
    t = compare_table(res)
    t.to_csv(out / f"transform_compare_{tag}_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.pivot_table(index=["economy", "horizon", "model"], columns="set", values="rel_rw").round(3).to_string())
    print("BEGIN COMPARE TABLE CSV")
    print(t.to_csv(index=False, float_format="%.4f"), end="")
    print("END COMPARE TABLE CSV")
    print(f"wrote {out / f'transform_compare_{tag}.csv.gz'} and the table")


def cmd_pilot(args):
    from ..data import Downloader as DL
    from .pilot import run_pilot
    from .transform_search import compare_table

    store = load_frames(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_pilot(store, DL(cache_dir=args.cache, max_age_days=args.max_age), args.countries, tuple(args.horizons),
                    tuple(args.models), start=args.start, n_jobs=args.jobs, progress=_progress(t0),
                    bottomup=not args.no_bottom_up)
    pd.to_pickle(res, out / "pilot.pkl")
    head = {k: v for k, v in res.items() if not k[2].startswith("comp:")}
    table = compare_table(head)
    table.to_csv(out / "pilot_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        med = table[table["model"] == "median"]
        print(med.pivot_table(index=["economy", "horizon"], columns="set", values="rel_rw").round(3).to_string())
    print(f"wrote {out / 'pilot.pkl'} ({len(res)} forecast series) and pilot_table.csv")


def cmd_results(args):
    """Move results between output/ pickles and compact CSV files that are kept in the repository."""
    from .evaluation import pack_results, unpack_results

    out, store_dir = Path(args.out), Path(args.store)
    if args.action == "pack":
        store_dir.mkdir(parents=True, exist_ok=True)
        res = pd.read_pickle(out / f"{args.name}.pkl")
        if args.countries != CODES:
            res = {k: v for k, v in res.items() if k[0] in args.countries}
        path = store_dir / f"{args.tag or args.name}.csv.gz"
        n = pack_results(res, path)
        print(f"wrote {path} ({n} forecast series, {path.stat().st_size / 1e6:.2f} MB)")
        return
    files = sorted(store_dir.glob("*.csv.gz"))
    groups = {"results": [], "deep": [], "foundation": []}
    for f in files:
        kind = "deep" if f.name.startswith("deep") else "foundation" if f.name.startswith("foundation") else "results"
        groups[kind].append(f)
    out.mkdir(parents=True, exist_ok=True)
    for kind, fs in groups.items():
        if fs:
            res = unpack_results(fs)
            pd.to_pickle(res, out / f"{kind}.pkl")
            print(f"wrote {out / f'{kind}.pkl'} ({len(res)} forecast series from {len(fs)} file(s))")


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
    t.add_argument("--panel-only", action="store_true", help="run only the pooled models (needs all four economies)")
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
    rep.add_argument("--latest-only", action="store_true", help="only refresh the latest-forecast section of --out")
    rep.set_defaults(func=cmd_report)

    f = sub.add_parser("forecast", parents=[common])
    f.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--out", default="output/inflation")
    f.set_defaults(func=cmd_forecast)

    sc = sub.add_parser("scenarios", parents=[common], help="what-if responses of the latest forecasts")
    sc.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    sc.add_argument("--jobs", type=int, default=1)
    sc.add_argument("--resume", action="store_true", help="skip model fits already in the output file")
    sc.add_argument("--out", default="output/inflation")
    sc.set_defaults(func=cmd_scenarios)

    tr = sub.add_parser("transforms", parents=[common], help="search over predictor transformations")
    tr.add_argument("stage", choices=["screen", "compare", "report"])
    tr.add_argument("--report", default="docs/inflation_transforms.md")
    tr.add_argument("--cache", default=None)
    tr.add_argument("--max-age", type=float, default=7.0, help="days before cached downloads are refreshed")
    tr.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    tr.add_argument("--sets", nargs="+", default=["current", "core_anchor", "top1", "top3", "library", "lags"])
    tr.add_argument("--models", nargs="+", default=["ridge", "adalasso", "enet", "krr", "rf", "lgbm"])
    tr.add_argument("--jobs", type=int, default=1)
    tr.add_argument("--out", default="output/inflation")
    tr.set_defaults(func=cmd_transforms)

    pl = sub.add_parser("pilot", parents=[common], help="nowcast, survey path and bottom-up pilots")
    pl.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12, 24])
    pl.add_argument("--models", nargs="+", default=["ridge", "adalasso", "enet", "krr", "rf", "lgbm"])
    pl.add_argument("--start", default="2000-01")
    pl.add_argument("--jobs", type=int, default=1)
    pl.add_argument("--no-bottom-up", action="store_true")
    pl.add_argument("--cache", default=None)
    pl.add_argument("--max-age", type=float, default=7.0)
    pl.add_argument("--out", default="output/inflation/pilot")
    pl.set_defaults(func=cmd_pilot)

    rs = sub.add_parser("results", parents=[common], help="pack results into results/inflation or unpack them")
    rs.add_argument("action", choices=["pack", "unpack"])
    rs.add_argument("--name", default="results", help="pickle to pack: results, deep or foundation")
    rs.add_argument("--tag", default=None, help="file name to write (default: --name)")
    rs.add_argument("--out", default="output/inflation")
    rs.add_argument("--store", default="results/inflation")
    rs.set_defaults(func=cmd_results)

    st = sub.add_parser("site", parents=[common], help="interactive site with forecasts and scenarios")
    st.add_argument("--results", default="output/inflation")
    st.add_argument("--scores", default="docs/inflation_report.csv", help="league table written by `report`")
    st.add_argument("--template", default="docs/site/template.html")
    st.add_argument("--out", default="docs/site/index.html")
    st.add_argument("--previous", default=None, help="page whose backtest to reuse when results.pkl is missing (default: --out)")
    st.set_defaults(func=cmd_site)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
