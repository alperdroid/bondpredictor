"""Command line interface.

    bondpredictor build      download data and write the monthly panel
    bondpredictor insample   predictive regressions with Hodrick/NW/bootstrap
    bondpredictor backtest   real-time out-of-sample evaluation of the model zoo
    bondpredictor forecast   latest forecasts from one model
    bondpredictor termpremium  ACM term premium for a country
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .data import Downloader


def _dataset(args):
    from .dataset import build_dataset

    dl = Downloader(cache_dir=args.cache, offline=args.offline, max_age_days=args.max_age)
    return build_dataset(tuple(args.countries), dl=dl, end=getattr(args, "end", None))


def _load_frame(args) -> pd.DataFrame:
    path = Path(args.data) if getattr(args, "data", None) else None
    if path and path.exists():
        frame = pd.read_pickle(path)
        keep = [c for c in frame.columns if c.split(".")[0] in set(args.countries) | {"G"}]
        return frame[keep]
    return _dataset(args).frame


def cmd_build(args):
    ds = _dataset(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.frame.to_pickle(out)
    ds.frame.to_csv(out.with_suffix(".csv"))
    print(f"wrote {out} ({ds.frame.shape[0]} months x {ds.frame.shape[1]} columns)")
    for code, d in ds.countries.items():
        y10 = d.frame["y_10"].dropna()
        print(f"\n{code}: curve={d.curve_source}  10y yield {y10.index.min()}..{y10.index.max()}")
        for note in d.notes:
            print(f"  - {note[:160]}")


def cmd_insample(args):
    from .inference import null_bootstrap, predictive_regression
    from .learned import PCAFactors
    from .pipeline import feature_blocks
    from .returns import rx_column

    frame = _load_frame(args)
    c = args.country
    blocks = feature_blocks(c, args.countries, frame)
    names = args.blocks or ["slope", "cp", "global", "cycle", "carry"]
    cols, trs = [], []
    for n in names:
        cols += blocks[n][0]
        trs += blocks[n][1]
    data = frame
    for tr in trs:
        data = pd.concat([data, tr.clone().fit_transform(data)], axis=1)
        data = data.loc[:, ~data.columns.duplicated(keep="last")]
    cols = [x for x in dict.fromkeys(cols) if x in data and data[x].notna().sum() > 60]
    y = data[f"{c}.{rx_column(args.horizon, args.maturity)}"]
    r1 = data[f"{c}.{rx_column(1, args.maturity)}"]
    X = data[cols]
    if args.start:
        X = X.loc[pd.Period(args.start, "M") :]
    res = predictive_regression(y, X, args.horizon, one_period=r1)
    print(f"{c}: {args.horizon}-month excess return on the {args.maturity:g}-year bond\n")
    print(res)
    if args.bootstrap:
        p = null_bootstrap(X, r1, args.horizon, res, n_boot=args.bootstrap)
        print("\nBootstrap p-values under no predictability (persistent predictors):")
        print(p.to_string(float_format=lambda v: f"{v:.3f}"))


def cmd_backtest(args):
    from .pipeline import run_all

    frame = _load_frame(args)
    res = run_all(
        frame,
        list(args.countries),
        start=args.start,
        min_train=args.min_train,
        maturity=args.maturity,
        horizon=args.horizon,
        models=args.models,
        panel=not args.no_panel,
        exclude_regimes=args.exclude_regimes or (),
        ml=args.ml,
        n_jobs=args.jobs,
        window=args.window,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(out / "summary.csv", index=False)
    pd.to_pickle(res.forecasts, out / "forecasts.pkl")
    for (c, name), fc in res.forecasts.items():
        fc.to_csv(out / f"forecasts_{c}_{name.replace(':', '_').replace('+', '_')}.csv")
    show = res.summary.copy()
    cols = ["country", "model", "N", "start", "R2_OS", "CW_t", "CW_p", "sharpe_model", "sharpe_mean", "sharpe_hold", "cer_gain_bp"]
    show = show[[c for c in cols if c in show]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}/summary.csv and per-model forecasts")


def cmd_robustness(args):
    from .pipeline import run_jobs
    from .robustness import country_report, robustness_jobs

    frame = _load_frame(args)
    main = pd.read_pickle(Path(args.forecasts))
    variants: dict = {}
    saved = Path(args.out) / "robustness_variants.pkl"
    if args.reuse_variants and saved.exists():
        variants = pd.read_pickle(saved)
    elif not args.skip_variants:
        jobs = robustness_jobs(frame, list(args.countries), start=args.start, min_train=args.min_train)
        print(f"running {len(jobs)} re-estimation variants on {args.jobs} processes ...", flush=True)
        variants = run_jobs(frame, jobs, args.jobs)
        pd.to_pickle(variants, Path(args.out) / "robustness_variants.pkl")
    sections = []
    for c in args.countries:
        m = {name: r for (cc, name), r in main.items() if cc == c}
        v = {name: r for (cc, name), r in variants.items() if cc == c and not r.empty}
        sections.append(country_report(c, frame, m, v, args.horizon, n_boot=args.n_boot))
    text = "# Robustness report\n\n" + "\n\n".join(sections)
    path = Path(args.out) / "robustness.md"
    path.write_text(text)
    print(text)
    print(f"\nwrote {path}")


def cmd_forecast(args):
    from .backtest import run_backtest, run_panel_backtest
    from .pipeline import country_specs, panel_spec

    frame = _load_frame(args)
    last = frame.index[-1]
    rows = []
    for c in args.countries:
        if args.model.startswith("panel"):
            continue
        for spec in country_specs(c, args.countries, frame, args.maturity, args.horizon, [args.model]):
            r = run_backtest(frame, spec, start=str(last - 1), min_train=args.min_train)
            for d, row in r.iterrows():
                rows.append({"country": c, "date": d, **row.to_dict()})
    if args.model.startswith("panel"):
        spec = panel_spec(args.countries, frame, args.maturity, args.horizon)
        for c, r in run_panel_backtest(frame, spec, start=str(last - 1), min_train=args.min_train).items():
            for d, row in r.iterrows():
                rows.append({"country": c, "date": d, **row.to_dict()})
    out = pd.DataFrame(rows)
    if out.empty:
        print("no forecasts")
        return
    out = out[out["date"] == out["date"].max()] if not args.all else out
    print(f"Forecast of the {args.horizon}-month log excess return on the {args.maturity:g}-year bond ({args.model})")
    print(out[["country", "date", "forecast", "benchmark", "n_train", "features"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def cmd_termpremium(args):
    from .acm import acm
    from .dataset import build_country
    from . import config

    dl = Downloader(cache_dir=args.cache, offline=args.offline, max_age_days=args.max_age)
    cd = build_country(dl, config.COUNTRIES[args.country])
    if cd.curve is None:
        print(f"{args.country}: no zero curve available ({cd.curve_source}); ACM needs a curve")
        return 1
    short = None if cd.curve.yield_at(1 / 12).notna().mean() > 0.9 else cd.frame["short"]
    res = acm(cd.curve, short_rate=short, n_factors=args.factors, start=args.start)
    tp = (res.term_premium * 100).round(3)
    tp.columns = [f"tp_{c:g}y" for c in tp.columns]
    print(f"{args.country} ACM ({res.n_factors} factors, curve={cd.curve_source}), fit RMSE {res.fit_rmse_bp:.1f}bp")
    print(tp.tail(args.tail).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tp.to_csv(args.out)
        print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bondpredictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--countries", nargs="+", default=["US", "DE", "JP"], choices=["US", "DE", "JP"])
    common.add_argument("--cache", default=None, help="download cache directory (default data/cache)")
    common.add_argument("--offline", action="store_true", help="use cached downloads only")
    common.add_argument("--max-age", type=float, default=1.0, help="re-download cached files older than N days")
    common.add_argument("--data", default="data/processed/panel.pkl", help="prebuilt panel from `build` (used if it exists)")
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--maturity", type=float, default=10.0)
    model.add_argument("--horizon", type=int, default=12, choices=[1, 12], help="forecast horizon in months (1 only for the 10y bond)")
    model.add_argument("--min-train", type=int, default=120)

    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="download data and build the panel")
    b.add_argument("--out", default="data/processed/panel.pkl")
    b.add_argument("--end", default=None, help="last month to include (default: last complete month)")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("insample", parents=[common, model], help="in-sample predictive regressions")
    i.add_argument("--country", default="US", choices=["US", "DE", "JP"])
    i.add_argument("--blocks", nargs="+", default=None,
                   help="predictor blocks: slope cp global cycle carry macro value_mom risk supply_policy")
    i.add_argument("--start", default=None)
    i.add_argument("--bootstrap", type=int, default=0, help="bootstrap replications (0 = skip)")
    i.set_defaults(func=cmd_insample)

    t = sub.add_parser("backtest", parents=[common, model], help="out-of-sample evaluation")
    t.add_argument("--start", default="1990-01", help="first forecast origin")
    t.add_argument("--models", nargs="+", default=None, help="subset of the model zoo (e.g. slope cp curve all:ridge panel:ridge)")
    t.add_argument("--no-panel", action="store_true")
    t.add_argument("--exclude-regimes", nargs="+", default=None, help="drop training months in these regimes, e.g. ycc nirp")
    t.add_argument("--ml", action="store_true", help="also run the machine-learning models (needs scikit-learn)")
    t.add_argument("--jobs", type=int, default=1, help="parallel processes")
    t.add_argument("--window", type=int, default=None, help="rolling estimation window in months (default expanding)")
    t.add_argument("--out", default="output")
    t.set_defaults(func=cmd_backtest)

    r = sub.add_parser("robustness", parents=[common, model], help="robustness tests on backtest forecasts")
    r.add_argument("--forecasts", default="output/forecasts.pkl", help="forecasts.pkl written by `backtest`")
    r.add_argument("--start", default="1990-01")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--skip-variants", action="store_true", help="only tests on existing forecasts (no re-estimation)")
    r.add_argument("--reuse-variants", action="store_true", help="reuse robustness_variants.pkl from --out")
    r.add_argument("--out", default="output")
    r.set_defaults(func=cmd_robustness)

    f = sub.add_parser("forecast", parents=[common, model], help="latest forecast")
    f.add_argument("--model", default="slope")
    f.add_argument("--all", action="store_true")
    f.set_defaults(func=cmd_forecast)

    tp = sub.add_parser("termpremium", parents=[common], help="ACM term premium")
    tp.add_argument("--country", default="DE", choices=["US", "DE", "JP"])
    tp.add_argument("--factors", type=int, default=5)
    tp.add_argument("--start", default=None)
    tp.add_argument("--tail", type=int, default=12)
    tp.add_argument("--out", default=None)
    tp.set_defaults(func=cmd_termpremium)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
