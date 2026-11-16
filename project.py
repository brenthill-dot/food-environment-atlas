"""Food access analysis pipeline (ANLY 615, Milestone 3).

Business question (Milestone 2 memo):
    Which 10 Texas counties should the Texas Department of Agriculture and
    grocery retailers prioritize for food-access investment in 2027?

Steps:
    1. Query USDA Food Environment Atlas tables from PostgreSQL into
       Pandas with pd.read_sql() (no manual CSV exports).
    2. Pull Census County Business Patterns (CBP) grocery data (NAICS
       445110) from the Census API and stage it in the database.
    3. Clean, de-duplicate, merge (LEFT JOIN from USDA), engineer features.
    4. Score and rank counties; produce diagnostic and business charts.

Run:
    pip install pandas sqlalchemy psycopg2-binary requests matplotlib seaborn
    export CENSUS_API_KEY=your_free_key   # https://api.census.gov/data/key_signup.html
    python project.py
Optional: set DATABASE_URL to override the default local Docker database.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sqlalchemy import create_engine, text

SCHEMA = "atlas"
STATE_ABBR = "TX"
STATE_FIPS = "48"
OUTPUT_DIR = Path("figures")
CBP_URL = "https://api.census.gov/data/2023/cbp"
GROCERY_NAICS = "445110"

# (table, column prefix, required?) -- newest year suffix auto-detected.
MEASURES = [
    ("access", "laccess_pop", True),
    ("access", "pct_laccess_pop", True),
    ("access", "laccess_lowi", True),
    ("access", "laccess_hhnv", False),
    ("access", "laccess_snap", False),
    ("stores", "groc", False),
    ("stores", "grocpth", False),
    ("socioeconomic", "povrate", False),
    ("socioeconomic", "medhhinc", False),
]
SENTINELS = [-9999, -8888]  # Atlas "not available" codes

# Need indicators and weights for the priority score. Counts capture the
# size of the problem; shares/rates stop big counties winning on size alone.
# A county missing an indicator is scored on the indicators it has.
NEED_WEIGHTS = {
    "laccess_pop": 1.0,        # residents with low access (count)
    "laccess_lowi": 1.0,       # low-income, low-access residents (count)
    "pct_laccess_pop": 1.0,    # share of residents with low access
    "lowi_share": 1.0,         # low-income low-access share of population
    "hhnv_share": 1.0,         # no-vehicle low-access share of population
    "low_store_density": 1.0,  # fewer stores per 1,000 = higher need
}

CBP_COLUMNS = ["fips", "establishments", "employees", "payroll_thousands",
               "cbp_status"]


# ----------------------------------------------------------------------
# 1. Database extraction
# ----------------------------------------------------------------------
def get_engine():
    """Build a SQLAlchemy engine; defaults to the team Docker database."""
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://atlas_user:atlas_password"
        "@localhost:5432/food_atlas")
    return create_engine(url)


def latest_column(engine, table, prefix):
    """Return the newest-year column for a prefix, or None if absent."""
    query = text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = :t")
    cols = pd.read_sql(query, engine, params={"s": SCHEMA, "t": table})
    pattern = re.compile(rf"^{prefix}(\d{{2}})$")
    years = [(int(m.group(1)), c) for c in cols["column_name"]
             if (m := pattern.match(c))]
    return max(years)[1] if years else None


def load_atlas(engine):
    """Read the Atlas measures straight into DataFrames (one per table)."""
    wanted = {}
    for table, prefix, required in MEASURES:
        col = latest_column(engine, table, prefix)
        if col is None and required:
            raise SystemExit(f"No column like {prefix}## in {SCHEMA}.{table}")
        if col is None:
            print(f"Skipping optional measure {prefix}: not found")
            continue
        wanted.setdefault(table, []).append(col)

    frames = []
    for table, cols in wanted.items():
        extra = ["fips", "state", "county"] if table == "access" else ["fips"]
        query = (f"SELECT {', '.join(extra + cols)} FROM {SCHEMA}.{table} "
                 "WHERE fips IS NOT NULL")
        df = pd.read_sql(query, engine)
        print(f"Loaded {table}: {df.shape} -> {cols}")
        frames.append(df)
    return frames


# ----------------------------------------------------------------------
# 2. Census CBP via API (nested JSON -> DataFrame)
# ----------------------------------------------------------------------
def fetch_cbp():
    """Download Texas grocery (NAICS 445110) county rows from Census CBP.

    Tries several request variants and prints what the API returns, so a
    failure is easy to diagnose. Returns None if every attempt fails.
    """
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        print("No CENSUS_API_KEY set. Get a free key at "
              "https://api.census.gov/data/key_signup.html and run: "
              "export CENSUS_API_KEY=your_key")
    variants = [
        ("NAICS2022", "NAME,ESTAB,EMP,PAYANN,EMP_N,NAICS2022"),
        ("NAICS2022", "NAME,ESTAB,EMP,PAYANN,NAICS2022"),
        ("NAICS2017", "NAME,ESTAB,EMP,PAYANN,NAICS2017"),
    ]
    for naics_var, fields in variants:
        params = {
            "get": fields,
            "for": "county:*",
            "in": f"state:{STATE_FIPS}",
            naics_var: GROCERY_NAICS,
        }
        if key:
            params["key"] = key
        resp = requests.get(CBP_URL, params=params, timeout=60)
        print(f"Census try {naics_var}: status {resp.status_code}, "
              f"first 150 chars: {resp.text[:150]!r}")
        if resp.ok and resp.text.strip().startswith("["):
            rows = resp.json()  # list of lists; first row is the header
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df = df.rename(columns={naics_var: "naics_code"})
            if "EMP_N" not in df:
                df["EMP_N"] = None
            return df
    print("Census data unavailable; continuing without it.")
    return None


def clean_cbp(raw):
    """Build 5-char FIPS, cast text to numbers, flag suppressed cells."""
    df = raw.copy()
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df = df.drop_duplicates(subset=["fips", "naics_code"])
    for col in ("ESTAB", "EMP", "PAYANN"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # EMP_N is a noise/suppression flag; letter codes mean withheld values.
    df["cbp_status"] = df["EMP_N"].apply(
        lambda f: "suppressed" if isinstance(f, str) and f.strip()
        and f.strip() not in ("0", "None") else "reported")
    df = df.rename(columns={"ESTAB": "establishments", "EMP": "employees",
                            "PAYANN": "payroll_thousands"})
    return df[CBP_COLUMNS]


def empty_cbp():
    """Empty CBP table with correct column types (used if the API fails)."""
    return pd.DataFrame({
        "fips": pd.Series(dtype=str),
        "establishments": pd.Series(dtype=float),
        "employees": pd.Series(dtype=float),
        "payroll_thousands": pd.Series(dtype=float),
        "cbp_status": pd.Series(dtype=str),
    })


# ----------------------------------------------------------------------
# 3. Wrangling and features
# ----------------------------------------------------------------------
def clean_numeric(series):
    """Strip text noise (commas, %, $, footnote marks) and cast to float."""
    cleaned = (series.astype(str)
               .str.replace(r"[,$%*]", "", regex=True)
               .str.strip()
               .replace({"": None, "nan": None, "None": None, "NA": None}))
    return pd.to_numeric(cleaned, errors="coerce")


def wrangle_atlas(frames):
    """Clean each Atlas table, remove duplicates, merge on FIPS."""
    cleaned = []
    for df in frames:
        df = df.copy()
        df["fips"] = df["fips"].astype(str).str.strip().str.zfill(5)
        dupes = df.duplicated(subset="fips").sum()
        print(f"Duplicate county rows removed: {dupes}")
        df = df.drop_duplicates(subset="fips", keep="first")
        for col in df.columns.difference(["fips", "state", "county"]):
            df[col] = clean_numeric(df[col])
        # Diagnose WHY values are missing: blank cells vs "n/a" codes.
        texas = df[df["fips"].str.startswith(STATE_FIPS)]
        for col in df.columns.difference(["fips", "state", "county"]):
            n_code = texas[col].isin(SENTINELS).sum()
            n_blank = texas[col].isna().sum()
            print(f"  Texas {col}: {n_blank} blank, {n_code} coded n/a")
        df = df.replace(SENTINELS, np.nan)
        # Drop the year suffix so later code is year-agnostic.
        df.columns = [re.sub(r"\d{2}$", "", c) for c in df.columns]
        cleaned.append(df)

    merged = cleaned[0]
    for df in cleaned[1:]:
        merged = pd.merge(merged, df, on="fips", how="left")
    merged = merged[merged["state"] == STATE_ABBR].copy()
    print(f"Texas counties: {len(merged)} (expected 254)")
    return merged


def handle_missing(df, cbp):
    """Diagnose gaps; LEFT JOIN CBP and flag why a county has no data."""
    print("\nMissing share per column (Atlas):")
    print(df.isna().mean().sort_values(ascending=False).round(3))

    # Only the access measures are required. Counties with missing store
    # counts are KEPT and flagged: dropping them removed about half of
    # Texas (mostly rural counties) and would bias the ranking.
    core = ["laccess_pop", "laccess_lowi", "pct_laccess_pop"]
    before = len(df)
    df = df.dropna(subset=core).copy()
    print(f"Dropped {before - len(df)} counties missing core measures")

    for col in ("groc", "grocpth", "laccess_hhnv"):
        if col not in df:
            df[col] = np.nan
    df["stores_missing"] = df["grocpth"].isna()
    print(f"Counties with missing store data (kept, flagged): "
          f"{int(df['stores_missing'].sum())}")

    # No-vehicle counts: median-impute and flag the rows.
    df["hhnv_imputed"] = df["laccess_hhnv"].isna()
    df["laccess_hhnv"] = df["laccess_hhnv"].fillna(
        df["laccess_hhnv"].median())

    # LEFT JOIN keeps counties with no grocery establishments in Census.
    df = pd.merge(df, cbp, on="fips", how="left")
    df["cbp_status"] = df["cbp_status"].fillna("no_cbp_record")
    # A missing CBP row is NOT assumed to be zero; counts stay NaN.
    return df


def engineer_features(df):
    """Add population, shares, store density and a priority score."""
    df = df.copy()
    pct = df["pct_laccess_pop"].replace(0, np.nan) / 100
    df["population"] = (df["laccess_pop"] / pct).round()
    df["groc_per_1000"] = df["groc"] / df["population"] * 1000
    df["lowi_share"] = df["laccess_lowi"] / df["population"] * 100
    df["hhnv_share"] = df["laccess_hhnv"] / df["population"] * 100
    # Lower density of stores = greater need (NaN stays NaN, not penalized).
    df["low_store_density"] = -df["grocpth"]
    df["lowi_per_store"] = df["laccess_lowi"] / (df["groc"] + 1)
    df["payroll_per_estab"] = (df["payroll_thousands"]
                               / df["establishments"])

    # Priority score: weighted mean percentile rank over available metrics.
    ranks = df[list(NEED_WEIGHTS)].rank(pct=True)
    df["priority_score"] = weighted_score(ranks, NEED_WEIGHTS)
    df["priority_rank"] = df["priority_score"].rank(
        ascending=False, method="first").astype(int)

    # Sensitivity check: triple the weight on low-income residents.
    alt_weights = dict(NEED_WEIGHTS, laccess_lowi=3.0, lowi_share=3.0)
    alt = weighted_score(ranks, alt_weights)
    df["alt_rank"] = alt.rank(ascending=False, method="first").astype(int)
    return df.sort_values("priority_rank")


def weighted_score(ranks, weights):
    """Weighted mean of rank columns, ignoring missing values."""
    w = pd.Series(weights)
    total = ranks.mul(w, axis=1).sum(axis=1)
    available = ranks.notna().mul(w, axis=1).sum(axis=1)
    return total / available


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def diagnostic_plots(df):
    """Histogram and boxplots to check distributions and outliers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["groc_per_1000"].dropna(), bins=30, ax=axes[0])
    axes[0].set_title("Grocery stores per 1,000 residents")
    sns.histplot(df["lowi_share"].dropna(), bins=30, ax=axes[1])
    axes[1].set_title("Low-income, low-access share of population (%)")
    sns.boxplot(x=df["laccess_pop"], ax=axes[2])
    axes[2].set_title("Low-access population (outlier check)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic_distributions.png", dpi=200)
    plt.close(fig)


def stakeholder_charts(df):
    """Top-10 priority bar chart, need-vs-supply scatter, CBP view."""
    top10 = df.head(10).sort_values("priority_score")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top10["county"], top10["priority_score"], color="#1f5f8b")
    ax.set_title("Top 10 Texas counties by food-access priority score")
    ax.set_xlabel("Priority score (0-1, higher = greater need)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_priority_counties.png", dpi=200)
    plt.close(fig)

    plot_df = df.dropna(subset=["grocpth"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x="grocpth", y="lowi_share",
                    hue=plot_df["priority_rank"] <= 10, alpha=0.7, ax=ax)
    ax.set_title("Store availability vs. low-income, low-access share")
    ax.set_xlabel("Grocery stores per 1,000 residents")
    ax.set_ylabel("Low-income, low-access share of population (%)")
    ax.legend(title="Top 10 priority")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "need_vs_supply.png", dpi=200)
    plt.close(fig)

    top = df.head(10).dropna(subset=["establishments"])
    if not top.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(top["county"], top["establishments"], color="#2f8f5b")
        ax.set_title("Grocery establishments (CBP 2023) in top-priority "
                     "counties")
        ax.set_ylabel("Establishments")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "top10_cbp_establishments.png", dpi=200)
        plt.close(fig)


def main():
    """Run the full pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    engine = get_engine()
    atlas = wrangle_atlas(load_atlas(engine))

    raw_cbp = fetch_cbp()
    if raw_cbp is not None:
        cbp = clean_cbp(raw_cbp)
        cbp.to_sql("cbp_grocery_2023", engine, schema=SCHEMA,
                   if_exists="replace", index=False)  # stage in the database
        cbp = pd.read_sql(f"SELECT * FROM {SCHEMA}.cbp_grocery_2023", engine)
    else:
        cbp = empty_cbp()

    df = engineer_features(handle_missing(atlas, cbp))
    diagnostic_plots(df)
    stakeholder_charts(df)

    cols = ["priority_rank", "county", "population", "laccess_pop",
            "pct_laccess_pop", "laccess_lowi", "lowi_share", "grocpth",
            "stores_missing", "establishments", "cbp_status", "alt_rank"]
    df.head(10)[cols].to_csv(OUTPUT_DIR / "top10_table.csv", index=False)
    overlap = len(set(df.head(10)["county"])
                  & set(df.nsmallest(10, "alt_rank")["county"]))
    print(df.head(10)[cols].to_string(index=False))
    print(f"\nTop-10 overlap under alternative weights: {overlap}/10")
    print(df["cbp_status"].value_counts())


if __name__ == "__main__":
    main()