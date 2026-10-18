"""Where every inflation input comes from, economy by economy.

Each economy has a headline CPI (the forecast target), core and component
indices, and a set of predictors.  A predictor names one or more sources in
priority order: the first is used where it exists and older or later
sources fill the remaining months, rescaled at the first overlap (indices)
or taken as is (rates).  ``Series.lag`` is the publication lag in months: a
value for reference month ``m`` is treated as known at the end of month
``m + lag``.  CPI has lag 1 everywhere, so a forecast made at the end of
month ``k`` uses CPI up to ``k - 1``.

Predictor groups are the "hemispheres" of the hemisphere neural net and the
blocks of the report:

* ``expect``: surveys of price expectations and price-setting intentions
* ``slack``: labour market and real activity
* ``cost``: energy, commodities, producer and import prices, exchange rates
* ``money``: interest rates, money, credit and asset prices

Own-inflation features (``own``), the global inflation factor (``global``)
and the tax calendar (``calendar``) are built in ``features.py``.

Sources: FRED, OECD SDMX, ECB, Eurostat, the BoJ API, ONS and the New York
Fed.  The Bank of England was not reachable from the build sandbox, so the UK
has no inflation-expectations survey (the EC survey dropped the UK after
Brexit; the BoE/Ipsos survey is on the Bank's site), and ONS suspended its
producer price index after January 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Series, fred

OECD_CPI = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/{area}.M.N.CPI.IX.{item}.N._Z"
OECD_CPI18 = "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,1.0/{area}.M.N.CPI.IX.{item}.N._Z"
OECD_KEI = "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/{area}.M.{measure}.{unit}.{activity}.{adj}._Z"
OECD_FIN = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/{area}.M.{measure}.{unit}._Z._Z._Z._Z.N"
HICP = "prc_hicp_minr?geo=EA&unit=I15&coicop18={item}"


def oecd(key: str, **kw) -> Series:
    return Series("oecd", key, **kw)


def ecb(key: str, **kw) -> Series:
    return Series("ecb", key, **kw)


def estat(key: str, **kw) -> Series:
    return Series("eurostat", key, **kw)


def boj(key: str, **kw) -> Series:
    return Series("boj", key, **kw)


def ons(path: str, **kw) -> Series:
    return Series("ons", path, **kw)


def oecd_cpi(area: str, item: str = "_T", coicop2018: bool = False, lag: int = 1) -> Series:
    return oecd((OECD_CPI18 if coicop2018 else OECD_CPI).format(area=area, item=item), lag=lag)


def kei(area: str, measure: str, unit: str = "IX", activity: str = "_Z", adj: str = "Y", lag: int = 1) -> Series:
    return oecd(OECD_KEI.format(area=area, measure=measure, unit=unit, activity=activity, adj=adj), lag=lag)


@dataclass(frozen=True)
class Predictor:
    sources: tuple[Series, ...]
    transforms: tuple[str, ...]  # see features.TRANSFORMS
    group: str
    splice: str = "ratio"  # "ratio" for indices and prices, "level" for rates and balances


def P(sources, transforms, group, splice="ratio") -> Predictor:
    if isinstance(sources, Series):
        sources = (sources,)
    return Predictor(tuple(sources), tuple(transforms), group, splice)


@dataclass(frozen=True)
class TaxEvent:
    """A legislated consumption-tax/VAT change, usable from the month it was announced."""

    effective: str  # first month with the new rate
    change: float  # rate change in pp, scaled by the economy's weight where only part is affected
    announced: str


@dataclass
class InflationCountry:
    code: str
    name: str
    cpi: tuple[Series, ...]  # headline index, not seasonally adjusted (the target)
    core: tuple[Series, ...] = ()
    components: dict[str, tuple[Series, ...]] = field(default_factory=dict)
    predictors: dict[str, Predictor] = field(default_factory=dict)
    tax_events: tuple[TaxEvent, ...] = ()
    oil_local: tuple[Series, ...] = ()  # crude oil price in local currency
    oecd_code: str = ""


# Crude oil, US$ per barrel: Brent (daily, 1987-) with WTI (monthly, 1946-) before.
BRENT = fred("DCOILBRENTEU", how="mean")
WTI = fred("WTISPLC")

# ---------------------------------------------------------------- United States
US = InflationCountry(
    code="US",
    name="United States",
    oecd_code="USA",
    cpi=(fred("CPIAUCNS", lag=1),),
    core=(fred("CPILFENS", lag=1),),
    components={
        "energy": (fred("CPIENGNS", lag=1),),
        "food": (fred("CPIUFDNS", lag=1),),
        "services": (fred("CUUR0000SAS", lag=1),),
        "goods": (fred("CUUR0000SACL1E", lag=1),),  # commodities less food and energy
        "shelter": (fred("CUUR0000SAH1", lag=1),),
    },
    oil_local=(BRENT, WTI),
    predictors={
        "mich": P(fred("MICH", lag=0), ["lvl", "d12"], "expect", "level"),
        "cleveland1y": P(fred("EXPINF1YR", lag=0), ["lvl"], "expect", "level"),
        "breakeven5y": P(fred("T5YIE", how="mean"), ["lvl", "d3"], "expect", "level"),
        "philly_prices_paid": P(fred("PPCDFSA066MSFRBPHI", lag=0), ["lvl"], "expect", "level"),
        "philly_prices_future": P(fred("PPFDFSA066MSFRBPHI", lag=0), ["lvl"], "expect", "level"),
        "unemp": P(fred("UNRATE", lag=1), ["lvl", "d12", "gap"], "slack", "level"),
        "payrolls": P(fred("PAYEMS", lag=1), ["yoy"], "slack"),
        "claims": P(fred("ICSA", how="mean"), ["yoy"], "slack"),
        "wages": P(fred("AHETPI", lag=1), ["yoy"], "slack"),
        "wage_tracker": P(fred("FRBATLWGT3MMAUMHWGO", lag=2), ["lvl"], "slack", "level"),
        "ip": P(fred("INDPRO", lag=1), ["yoy", "gap_log"], "slack"),
        "capu": P(fred("TCU", lag=1), ["lvl"], "slack", "level"),
        "retail": P(fred("RSAFS", lag=1), ["yoy"], "slack"),
        "housing_starts": P(fred("HOUST", lag=1), ["yoy"], "slack"),
        "sentiment": P(fred("UMCSENT", lag=0), ["lvl"], "slack", "level"),
        "ppi": P(fred("PPIACO", lag=1), ["yoy", "c3"], "cost"),
        "import_prices": P(fred("IR", lag=1), ["yoy"], "cost"),
        "commodities": P(fred("PALLFNFINDEXM", lag=1), ["yoy"], "cost"),
        "food_commodities": P(fred("PFOODINDEXM", lag=1), ["yoy"], "cost"),
        "dollar": P(fred("NBUSBIS", lag=1), ["yoy"], "cost"),
        "fedfunds": P(fred("FEDFUNDS", lag=0), ["lvl", "d12"], "money", "level"),
        "gs10": P(fred("GS10", lag=0), ["lvl"], "money", "level"),
        "term_spread": P(fred("T10Y3M", how="mean"), ["lvl"], "money", "level"),
        "credit_spread": P(fred("BAA10Y", how="mean"), ["lvl"], "money", "level"),
        "m2": P(fred("M2SL", lag=1), ["yoy"], "money"),
        "loans": P(fred("BUSLOANS", lag=1), ["yoy"], "money"),
        "stocks": P(fred("NASDAQCOM", how="mean"), ["yoy"], "money"),
        "house_prices": P(fred("CSUSHPINSA", lag=2), ["yoy"], "money"),
    },
)

# --------------------------------------------------------------- United Kingdom
UK = InflationCountry(
    code="UK",
    name="United Kingdom",
    oecd_code="GBR",
    cpi=(oecd_cpi("GBR"),),
    core=(oecd_cpi("GBR", "_TXCP01_NRG"),),
    components={
        "energy": (oecd_cpi("GBR", "CP045_0722"),),
        "food": (oecd_cpi("GBR", "CP01"),),
        "services": (oecd_cpi("GBR", "SERV"),),
        "goods": (oecd_cpi("GBR", "GD"),),
    },
    oil_local=(Series("fred", "DCOILBRENTEU", how="mean", divide=fred("DEXUSUK", how="mean")),
               Series("fred", "WTISPLC", divide=fred("EXUSUK"))),
    # VAT: 8->15% (June 1979 budget), 15->17.5% (March 1991 budget), the 2008-09
    # temporary cut to 15% and its reversal, 17.5->20% (June 2010 budget).
    tax_events=(
        TaxEvent("1979-07", 7.0, "1979-06"),
        TaxEvent("1991-04", 2.5, "1991-03"),
        TaxEvent("2008-12", -2.5, "2008-11"),
        TaxEvent("2010-01", 2.5, "2008-11"),
        TaxEvent("2011-01", 2.5, "2010-06"),
    ),
    predictors={
        "unemp": P((kei("GBR", "UNEMP", "PT_LF", "_T", lag=2), fred("LRHUTTTTGBM156S", lag=2)), ["lvl", "d12", "gap"], "slack", "level"),
        # Average weekly earnings, whole economy (ONS, 2000-), OECD manufacturing earnings before.
        "earnings": P((ons("/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/lms", lag=2),
                       kei("GBR", "H_EARN", "IX", "C", lag=2), fred("LCEAMN01GBM661S", lag=2)), ["yoy"], "slack"),
        "ip": P(kei("GBR", "PRVM", "IX", "BTE", lag=2), ["yoy", "gap_log"], "slack"),
        "retail": P(kei("GBR", "TOVM", "IX", "G47", lag=1), ["yoy"], "slack"),
        "business_conf": P(kei("GBR", "BCICP", "PB", "C", lag=0), ["lvl"], "slack", "level"),
        "consumer_conf": P(kei("GBR", "CCICP", "PB", lag=0), ["lvl"], "slack", "level"),
        # ONS output PPI (manufacturing, home sales); ONS suspended PPI after January 2025.
        "ppi": P((ons("/economy/inflationandpriceindices/timeseries/gb7s/mm22", lag=1), fred("GBRPPDMMINMEI", lag=1)),
                 ["yoy"], "cost"),
        "reer": P(fred("CCRETT01GBM661N", lag=1), ["yoy"], "cost"),
        "gbpusd": P(fred("DEXUSUK", how="mean"), ["yoy"], "cost"),
        "policy_rate": P(fred("IRSTCI01GBM156N", lag=1), ["lvl", "d12"], "money", "level"),
        "gilt10": P(fred("IRLTLT01GBM156N", lag=1), ["lvl"], "money", "level"),
        "money": P(kei("GBR", "MABM", lag=2), ["yoy"], "money"),
        "stocks": P(oecd(OECD_FIN.format(area="GBR", measure="SHARE", unit="IX"), lag=1), ["yoy"], "money"),
    },
)

# ------------------------------------------------------------------------ Japan
JP = InflationCountry(
    code="JP",
    name="Japan",
    oecd_code="JPN",
    # FRED's OECD copy stops in 2021; the OECD's COICOP 2018 table runs to date.
    cpi=(oecd_cpi("JPN", coicop2018=True), oecd_cpi("JPN")),
    core=(oecd_cpi("JPN", "_TXCP01_NRG", coicop2018=True), oecd_cpi("JPN", "_TXCP01_NRG")),
    components={
        "energy": (oecd_cpi("JPN", "CP045_0722", coicop2018=True),),
        "food": (oecd_cpi("JPN", "CP01", coicop2018=True),),
    },
    oil_local=(Series("fred", "DCOILBRENTEU", how="mean", times=fred("DEXJPUS", how="mean")),
               Series("fred", "WTISPLC", times=fred("EXJPUS"))),
    # Consumption tax: introduced at 3% (law passed Dec 1988), 5% (law Nov 1994),
    # 8% (law Aug 2012), 10% (date fixed by the Nov 2016 postponement law).
    tax_events=(
        TaxEvent("1989-04", 3.0, "1988-12"),
        TaxEvent("1997-04", 2.0, "1994-11"),
        TaxEvent("2014-04", 3.0, "2012-08"),
        TaxEvent("2019-10", 2.0, "2016-11"),
    ),
    predictors={
        "tankan_output_prices": P(boj("CO/TK99F0000614GCQ00000", lag=1), ["lvl", "d12"], "expect", "level"),
        "tankan_output_prices_fcst": P(boj("CO/TK99F0000614GCQ10000", lag=1), ["lvl"], "expect", "level"),
        "tankan_input_prices": P(boj("CO/TK99F0000615GCQ00000", lag=1), ["lvl"], "expect", "level"),
        "tankan_infl_outlook_1y": P(boj("CO/TK99F0000204HCQ00000", lag=1), ["lvl"], "expect", "level"),
        "unemp": P((kei("JPN", "UNEMP", "PT_LF", "_T", lag=1), fred("LRUN64TTJPM156S", lag=1)), ["lvl", "d12", "gap"], "slack", "level"),
        "employment": P(kei("JPN", "EMP", "PS", "_T", lag=1), ["yoy"], "slack"),
        "earnings": P(kei("JPN", "H_EARN", "IX", "C", lag=2), ["yoy"], "slack"),
        "ip": P((kei("JPN", "PRVM", "IX", "BTE", lag=1), fred("JPNPROINDMISMEI", lag=1)), ["yoy", "gap_log"], "slack"),
        "retail": P(kei("JPN", "TOVM", "IX", "G47", lag=1), ["yoy"], "slack"),
        "consumer_conf": P(kei("JPN", "CCICP", "PB", lag=0), ["lvl"], "slack", "level"),
        "ppi": P(boj("PR01/PRCG20_2200000000", lag=1), ["yoy", "c3"], "cost"),
        "import_prices": P(boj("PR01/PRCG20_2600000000", lag=1), ["yoy", "c3"], "cost"),
        "sppi": P(boj("PR02/PRCS20_5200000000", lag=1), ["yoy"], "cost"),
        "reer": P(fred("CCRETT01JPM661N", lag=1), ["yoy"], "cost"),
        "usdjpy": P(fred("DEXJPUS", how="mean"), ["yoy"], "cost"),
        "call_rate": P(fred("IRSTCI01JPM156N", lag=1), ["lvl", "d12"], "money", "level"),
        "jgb10": P(fred("IRLTLT01JPM156N", lag=1), ["lvl"], "money", "level"),
        "money": P(kei("JPN", "MABM", lag=2), ["yoy"], "money"),
        "stocks": P(oecd(OECD_FIN.format(area="JPN", measure="SHARE", unit="IX"), lag=1), ["yoy"], "money"),
    },
)

# -------------------------------------------------------------------- Euro area
EA = InflationCountry(
    code="EA",
    name="Euro area",
    oecd_code="EA20",
    # Eurostat HICP (ECOICOP 2, changing composition) from 1996, OECD's euro-area CPI before.
    cpi=(estat(HICP.format(item="TOTAL"), lag=1), oecd_cpi("EA20")),
    core=(estat(HICP.format(item="TOT_X_NRG_FOOD"), lag=1), ecb("ICP/M.U2.N.XEF000.4.INX", lag=1)),
    components={
        "energy": (estat(HICP.format(item="NRG"), lag=1),),
        "food": (estat(HICP.format(item="FOOD"), lag=1),),
        "services": (estat(HICP.format(item="SERV"), lag=1), ecb("ICP/M.U2.N.SERV00.4.INX", lag=1)),
        "goods": (estat(HICP.format(item="IGD_NNRG"), lag=1), ecb("ICP/M.U2.N.IGXE00.4.INX", lag=1)),
    },
    oil_local=(Series("fred", "DCOILBRENTEU", how="mean", divide=fred("DEXUSEU", how="mean")), BRENT, WTI),
    # German VAT 16->19% (Jan 2007) and the temporary 2020 cut, weighted by
    # Germany's ~27% share of the euro-area HICP.
    tax_events=(
        TaxEvent("2007-01", 0.81, "2005-11"),
        TaxEvent("2020-07", -0.81, "2020-06"),
        TaxEvent("2021-01", 0.81, "2020-06"),
    ),
    predictors={
        "consumer_price_exp": P(estat("ei_bsco_m?geo=EA21&indic=BS-PT-NY&s_adj=SA", lag=0), ["lvl", "d12"], "expect", "level"),
        "consumer_price_past": P(estat("ei_bsco_m?geo=EA21&indic=BS-PT-LY&s_adj=SA", lag=0), ["lvl"], "expect", "level"),
        "industry_price_exp": P(estat("ei_bsin_m_r2?geo=EA21&indic=BS-ISPE&s_adj=SA", lag=0), ["lvl", "d12"], "expect", "level"),
        "retail_price_exp": P(estat("ei_bsrt_m_r2?geo=EA21&indic=BS-RPBS&s_adj=SA", lag=0), ["lvl"], "expect", "level"),
        "unemp": P((estat("une_rt_m?geo=EA21&age=TOTAL&sex=T&s_adj=SA&unit=PC_ACT", lag=2), fred("LRHUTTTTEZM156S", lag=2)),
                   ["lvl", "d12", "gap"], "slack", "level"),
        "ip": P(kei("EA20", "PRVM", "IX", "BTE", lag=2), ["yoy", "gap_log"], "slack"),
        "retail": P(kei("EA20", "TOVM", "IX", "G47", lag=2), ["yoy"], "slack"),
        "esi": P(estat("ei_bssi_m_r2?geo=EA21&indic=BS-ESI-I&s_adj=SA", lag=0), ["lvl"], "slack", "level"),
        "consumer_conf": P(kei("EA20", "CCICP", "PB", lag=0), ["lvl"], "slack", "level"),
        "ppi": P(estat("sts_inppd_m?geo=EA21&indic_bt=PRC_PRR_DOM&nace_r2=B-E36&s_adj=NSA&unit=I21", lag=2), ["yoy"], "cost"),
        "import_prices": P(estat("sts_inppnd_m?geo=EA21&indic_bt=PRC_PRR_NDOM&nace_r2=B-E36&s_adj=NSA&unit=I21", lag=2), ["yoy"], "cost"),
        "neer": P(ecb("EXR/M.E03.EUR.EN00.A", lag=0), ["yoy"], "cost"),
        "eurusd": P(fred("DEXUSEU", how="mean"), ["yoy"], "cost"),
        "euribor3m": P(ecb("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", lag=0), ["lvl", "d12"], "money", "level"),
        "bond10": P((ecb("IRS/M.I9.L.L40.CI.0000.EUR.N.Z", lag=1), fred("IRLTLT01EZM156N", lag=1)), ["lvl"], "money", "level"),
        "m3": P(ecb("BSI/M.U2.Y.V.M30.X.1.U2.2300.Z01.E", lag=1), ["yoy"], "money"),
    },
)

COUNTRIES: dict[str, InflationCountry] = {"US": US, "UK": UK, "JP": JP, "EA": EA}

# Global block: commodity prices (US$) and supply-chain pressure, shared by every economy.
GLOBAL_PREDICTORS = {
    "commodities": P(fred("PALLFNFINDEXM", lag=1), ["yoy"], "global"),
    "food_commodities": P(fred("PFOODINDEXM", lag=1), ["yoy"], "global"),
    # New York Fed Global Supply Chain Pressure Index (1998-), published early the next month.
    "gscpi": P(Series("nyfed", "gscpi", lag=1), ["lvl", "d3"], "cost", "level"),
}

# Advanced economies whose median inflation is the global inflation factor
# (Ciccarelli-Mojon 2010).  OECD CPIs are published about six weeks after the
# reference month, hence lag 2.
GLOBAL_FACTOR_AREAS = ("AUT", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "GRC",
                       "IRL", "ITA", "JPN", "KOR", "NLD", "NOR", "PRT", "SWE", "USA")
GLOBAL_FACTOR_LAG = 2

# Countries excluded from the panel that trains the global deep-learning
# models: hyperinflation episodes or too-short monthly histories.
DEEP_PANEL_EXCLUDE = ("TUR", "ARG", "BRA", "RUS", "AUS", "G20", "G7", "OECD", "OECDE", "EU27_2020", "EA", "EA20")
