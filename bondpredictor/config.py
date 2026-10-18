"""Where every input comes from, country by country.

Each :class:`Series` names a public source, a publication lag in months, and
how to aggregate it to month-end.  The lags are conservative.  A value for
reference month ``m`` is treated as known at the end of ``m + lag``, which
approximates real-time availability with revised data.  Swap any entry for
``Series("csv", "/path/to/file.csv")`` to use your own file.

Known gaps in the free data (October 2026):

* FRED's OECD copies of Japanese CPI stop in mid-2021 and Japanese IP in
  early 2024.  After that, Japanese inflation is extended with the GDP
  deflator unless a local CPI file is configured.
* German IP from the OECD stops in early 2024 (the panel simply drops it).
* Implied rate volatility (MOVE, swaptions) is paid; realised volatility of
  daily yield changes is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Series:
    source: str  # "fred" | "bundesbank" | "oecd" | "ecb" | "eurostat" | "boj" | "ons" | "nyfed" | "csv"
    key: str
    lag: int = 0  # publication lag, months
    how: str = "last"  # month aggregation for daily/weekly data: "last" | "mean"
    tcode: int = 1  # FRED-MD style transformation (macro panel only)
    kind: str = "level"  # "level" or "yoy_pct" (already a % change on a year ago)
    minus: "Series | None" = None  # value = self - minus (e.g. spreads)
    divide: "Series | None" = None  # value = self / divide * scale
    times: "Series | None" = None  # value = self * times * scale
    scale: float = 1.0


def fred(key: str, **kw) -> Series:
    return Series("fred", key, **kw)


def bbk(key: str, **kw) -> Series:
    return Series("bundesbank", key, **kw)


@dataclass
class CountryConfig:
    code: str
    name: str
    curve_sources: tuple[str, ...]
    # Funding rate for one-month returns.  Either a maturity read off the
    # zero curve, or a money-market series in percent.
    short_rate_from_curve: float | None = None
    short_rate: Series | None = None
    short_rate_convention: str = "cc"  # "cc" | "discount3m" | "simple3m" | "overnight"
    long_yield_daily: Series | None = None  # for realised volatility
    par_long: Series | None = None  # 10y par yield for the par-bond fallback
    cpi: Series | None = None
    cpi_extension: Series | None = None
    core_cpi: Series | None = None
    core_cpi_extension: Series | None = None
    real_gdp: Series | None = None
    cb_assets_gdp: Series | None = None
    debt_gdp: Series | None = None
    use_fredmd: bool = False
    macro_panel: dict[str, Series] = field(default_factory=dict)
    extras: dict[str, Series] = field(default_factory=dict)
    regimes: dict[str, tuple[str, str]] = field(default_factory=dict)
    maturities: tuple[float, ...] = (1, 2, 3, 4, 5, 7, 10)


# ---------------------------------------------------------------- United States
US_GDP = fred("GDP", lag=1)
US = CountryConfig(
    code="US",
    name="United States",
    curve_sources=("gsw", "kim_wright"),
    short_rate=fred("DTB3"),
    short_rate_convention="discount3m",
    long_yield_daily=fred("DGS10"),
    par_long=fred("DGS10"),
    cpi=fred("CPIAUCSL", lag=1),
    core_cpi=fred("CPILFESL", lag=1),
    real_gdp=fred("GDPC1", lag=1),
    # Fed holdings of Treasuries / GDP (both $bn, quarterly since 1970).
    cb_assets_gdp=fred("FDHBFRBN", lag=1, divide=US_GDP),
    # Marketable debt held by the public / GDP ($mn / $bn).
    debt_gdp=fred("FYGFDPUN", lag=2, divide=US_GDP, scale=1e-3),
    use_fredmd=True,
    macro_panel={
        "indpro": fred("INDPRO", lag=1, tcode=5),
        "capu": fred("CUMFNS", lag=1, tcode=2),
        "unrate": fred("UNRATE", lag=1, tcode=2),
        "payems": fred("PAYEMS", lag=1, tcode=5),
        "emp": fred("CE16OV", lag=1, tcode=5),
        "hours": fred("AWHMAN", lag=1, tcode=1),
        "claims": fred("ICSA", how="mean", tcode=5),
        "houst": fred("HOUST", lag=1, tcode=4),
        "permit": fred("PERMIT", lag=1, tcode=4),
        "retail": fred("RSAFS", lag=1, tcode=5),
        "income": fred("DSPIC96", lag=1, tcode=5),
        "pce": fred("DPCERA3M086SBEA", lag=1, tcode=5),
        "m2": fred("M2SL", lag=1, tcode=6),
        "loans": fred("BUSLOANS", lag=1, tcode=6),
        "cpi": fred("CPIAUCSL", lag=1, tcode=6),
        "core": fred("CPILFESL", lag=1, tcode=6),
        "ppi": fred("PPIACO", lag=1, tcode=6),
        "pcepi": fred("PCEPI", lag=1, tcode=6),
        "oil": fred("WTISPLC", tcode=5),
        "ff": fred("FEDFUNDS", tcode=2),
        "tb3": fred("TB3MS", tcode=2),
        "term": fred("T10Y3M", tcode=1),
        "baa": fred("BAA10Y", tcode=1),
        "aaa": fred("AAA10Y", tcode=1),
        "jpy": fred("EXJPUS", tcode=5),
        "gbp": fred("EXUSUK", tcode=5),
        "nasdaq": fred("NASDAQCOM", tcode=5),
        "sent": fred("UMCSENT", lag=1, tcode=2),
    },
    extras={
        "credit": fred("BAA10Y"),
        "infl_exp": fred("MICH", lag=1),
    },
    regimes={
        "qe": ("2008-11", "2014-10"),
        "qe2020": ("2020-03", "2022-03"),
    },
    maturities=(1, 2, 3, 4, 5, 7, 10),
)

# ---------------------------------------------------------------------- Germany
EA_GDP = fred("EUNNGDP", lag=2)  # euro area nominal GDP, EUR mn per quarter
DE_10Y_OECD = fred("IRLTLT01DEM156N")
DE = CountryConfig(
    code="DE",
    name="Germany",
    curve_sources=("bundesbank",),
    short_rate_from_curve=0.25,
    short_rate=fred("IR3TIB01DEM156N"),  # used only if the curve lacks the short end
    short_rate_convention="simple3m",
    long_yield_daily=bbk("BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"),
    par_long=DE_10Y_OECD,
    cpi=fred("DEUCPIALLMINMEI", lag=1),  # OECD CPI, ends 2025-03
    cpi_extension=fred("CP0000DEM086NEST", lag=1),  # HICP (Eurostat)
    core_cpi=fred("CPGRLE01DEM659N", lag=1, kind="yoy_pct"),
    core_cpi_extension=fred("00XEFDDEM086NEST", lag=1),  # HICP ex energy, food, alcohol, tobacco
    real_gdp=fred("CLVMNACSCAB1GQDE", lag=2),
    # Eurosystem total assets / euro-area annual GDP.
    cb_assets_gdp=fred("ECBASSETSW", divide=EA_GDP, scale=0.25),
    debt_gdp=fred("GGGDTADEA188N", lag=4),
    macro_panel={
        "long": fred("IRLTLT01DEM156N", tcode=2),
        "mm3": fred("IR3TIB01DEM156N", tcode=2),
        "call": fred("IRSTCI01DEM156N", tcode=2),
        "unemp": fred("LRHUTTTTDEM156S", lag=1, tcode=2),
        "cpi": fred("DEUCPIALLMINMEI", lag=1, tcode=8),
        "hicp": fred("CP0000DEM086NEST", lag=1, tcode=8),
        "core": fred("00XEFDDEM086NEST", lag=1, tcode=8),
        "ip": fred("DEUPRINTO01IXOBM", lag=2, tcode=8),
        "stocks": fred("SPASTT01DEM661N", tcode=5),
        "reer": fred("CCRETT01DEM661N", lag=1, tcode=5),
        "exports": fred("XTEXVA01DEM664S", lag=2, tcode=5),
        "bconf": fred("BSCICP02DEM460S", lag=1, tcode=2),
        "cconf": fred("CSCICP02DEM460S", lag=1, tcode=2),
        "ecb": fred("ECBASSETSW", tcode=5),
        "eur": fred("DEXUSEU", tcode=5),
    },
    extras={
        # Safe-haven demand: Italy minus Germany 10y (OECD monthly averages).
        "periphery": fred("IRLTLT01ITM156N", minus=DE_10Y_OECD),
    },
    regimes={
        "nirp": ("2014-06", "2022-07"),
        "app": ("2015-03", "2022-06"),
    },
    maturities=(1, 2, 3, 4, 5, 7, 10),
)

# ------------------------------------------------------------------------ Japan
JP_NGDP = fred("JPNNGDP", lag=2)
JP_RGDP = fred("JPNRGDPEXP", lag=2)
JP_DEFLATOR = fred("JPNNGDP", lag=2, divide=JP_RGDP)
JP = CountryConfig(
    code="JP",
    name="Japan",
    curve_sources=("mof", "par_fred"),
    short_rate=fred("IRSTCI01JPM156N"),  # uncollateralised overnight call rate
    short_rate_convention="overnight",
    long_yield_daily=None,  # filled from MOF daily par yields when available
    par_long=fred("IRLTLT01JPM156N"),
    cpi=fred("JPNCPIALLMINMEI", lag=1),  # ends 2021-06 on FRED
    cpi_extension=JP_DEFLATOR,
    core_cpi=fred("CPGRLE01JPM659N", lag=1, kind="yoy_pct"),  # ends 2021-06 on FRED
    core_cpi_extension=JP_DEFLATOR,
    real_gdp=JP_RGDP,
    # BoJ total assets (100mn yen) / nominal GDP (bn yen, SAAR).
    cb_assets_gdp=fred("JPNASSETS", lag=1, divide=JP_NGDP, scale=0.1),
    debt_gdp=fred("GGGDTAJPA188N", lag=4),
    macro_panel={
        "long": fred("IRLTLT01JPM156N", tcode=2),
        "call": fred("IRSTCI01JPM156N", tcode=2),
        "unemp": fred("LRUN64TTJPM156S", lag=1, tcode=2),
        "cpi": fred("JPNCPIALLMINMEI", lag=1, tcode=8),
        "ip": fred("JPNPRINTO01IXOBM", lag=1, tcode=8),
        "stocks": fred("SPASTT01JPM661N", tcode=5),
        "reer": fred("CCRETT01JPM661N", lag=1, tcode=5),
        "exports": fred("XTEXVA01JPM664S", lag=2, tcode=5),
        "boj": fred("JPNASSETS", lag=1, tcode=5),
        "jpy": fred("DEXJPUS", tcode=5),
    },
    regimes={
        "qqe": ("2013-04", "2024-03"),
        "ycc": ("2016-09", "2024-03"),
        "nirp": ("2016-02", "2024-03"),
    },
    maturities=(1, 2, 3, 4, 5, 7, 10),
)

COUNTRIES: dict[str, CountryConfig] = {"US": US, "DE": DE, "JP": JP}

# Cross-country inputs.
GLOBAL = {
    "vix": fred("VIXCLS"),
}
