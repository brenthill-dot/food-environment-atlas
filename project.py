"""Food access analysis pipeline (ANLY 615, Milestone 3).

Business question (Milestone 2 memo):
    Which 10 Texas counties should the Texas Department of Agriculture and
    grocery retailers prioritize for food-access investment in 2027?

Steps:
    1. Query USDA Food Environment Atlas tables from PostgreSQL into
       Pandas with pd.read_sql() (no manual CSV exports).
    2. Pull Census County Business Patterns (CBP) grocery data (NAICS
       445110, county totals only) from the Census API and stage it in the
       keyed atlas.cbp_grocery_2023 table.
    3. Clean, de-duplicate, merge (LEFT JOIN from USDA), engineer features.
    4. Score all 254 counties on six need measures, rank them, and test the
       ranking with an alternative weighting; produce diagnostic and
       business charts.

Run:
    pip install pandas sqlalchemy psycopg2-binary requests matplotlib seaborn
    psql ... -f sql/02_keys_and_cbp.sql   (once; creates keys + CBP table)
    python food_access_pipeline.py
Optional: set DATABASE_URL to override the default local Docker database.
Optional: set CENSUS_API_KEY (used if present; recommended by Census).

Column names follow the Food Environment Atlas codebook. The newest year
available for each measure (e.g. groc20) is detected automatically.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sqlalchemy import create_engine, text

SCHEMA = "atlas"
STATE_ABBR = "TX"
STATE_FIPS = "48"
OUTPUT_DIR = Path("figures")
CBP_URL = "https://api.census.gov/data/2023/cbp"
CBP_TABLE = "cbp_grocery_2023"
GROCERY_NAICS = "445110"

# (table, column prefix, required?) -- newest year suffix auto-detected.
MEASURES = [
    ("access", "laccess_pop", True),
    ("access", "pct_laccess_pop", True),
    ("access", "laccess_lowi", True),
    ("access", "pct_laccess_lowi", True),
    ("access", "pct_laccess_hhnv", True),
    ("stores", "groc", True),
    ("stores", "grocpth", True),
    ("socioeconomic", "povrate", False),
    ("socioeconomic", "medhhinc", False),
]
SENTINELS = [-9999, -8888]  # Atlas "not available" codes

# The six need indicators in the priority score (higher = greater need).
NEED_METRICS = [
    "laccess_pop",        # low-access residents (count)
    "laccess_lowi",       # low-income, low-access residents (count)
    "pct_laccess_pop",    # low-access share of population
    "pct_laccess_lowi",   # low-income, low-access share
    "pct_laccess_hhnv",   # no-vehicle, low-access share
    "low_store_density",  # grocery stores per 1,000, sign flipped
]
# Main score: equal weights. Sensitivity check: triple the weight on the
# two low-income measures (count and share).
BASE_WEIGHTS = {m: 1 for m in NEED_METRICS}
ALT_WEIGHTS = {**BASE_WEIGHTS, "laccess_lowi": 3, "pct_laccess_lowi": 3}

# Access measures define who is in need, so a county without them cannot
# be scored. (In the 2025 Atlas no Texas county is missing them.)
CORE_ACCESS = ["laccess_pop", "laccess_lowi", "pct_laccess_pop",
               "pct_laccess_lowi"]


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
    """Download Texas grocery (NAICS 445110) county totals from Census CBP.

    EMPSZES=001 (all employment sizes) and LFO=001 (all legal forms)
    request one total row per county. Without them the API can return a
    row per size class, and a de-duplication step would keep an arbitrary
    one. If the LFO filter is not accepted, retry with EMPSZES only.
    """
    key = os.environ.get("CENSUS_API_KEY")
    totals_filters = [{"EMPSZES": "001", "LFO": "001"}, {"EMPSZES": "001"}]
    last_error = "no request sent"
    for naics_var in ("NAICS2017", "NAICS2022"):  # variable name by vintage
        for filters in totals_filters:
            params = {
                "get": f"NAME,ESTAB,EMP,PAYANN,EMP_N,{naics_var}",
                "for": "county:*",
                "in": f"state:{STATE_FIPS}",
                naics_var: GROCERY_NAICS,
                **filters,
            }
            if key:
                params["key"] = key
            resp = requests.get(CBP_URL, params=params, timeout=60)
            if resp.ok and resp.text.strip():
                rows = resp.json()  # list of lists; first row is the header
                df = pd.DataFrame(rows[1:], columns=rows[0])
                print(f"CBP: {len(df)} rows using {naics_var}, {filters}")
                return df.rename(columns={naics_var: "naics_code"})
            last_error = f"{resp.status_code} {resp.text[:200]}"
    raise SystemExit(f"Census API request failed: {last_error}")


def clean_cbp(raw):
    """Build 5-char FIPS, cast text to numbers, check one row per county."""
    df = raw.copy()
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    # County 999 is CBP's "statewide" bucket, not a real county.
    df = df[df["county"] != "999"]

    exact = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"CBP exact duplicate rows removed: {exact}")
    repeated = df["fips"].duplicated().sum()
    if repeated:
        raise SystemExit(f"{repeated} counties still have several CBP rows; "
                         "the county-total filter did not apply.")

    for col in ("ESTAB", "EMP", "PAYANN"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # EMP_N is the noise flag on employment (G = low, H = moderate,
    # J = high noise). The value is still reported, so keep the flag as
    # metadata instead of treating the row as missing.
    df["emp_noise_flag"] = df["EMP_N"].apply(
        lambda f: f.strip() if isinstance(f, str) and f.strip() else None)
    df = df.rename(columns={"ESTAB": "establishments", "EMP": "employees",
                            "PAYANN": "payroll_thousands"})
    return df[["fips", "establishments", "employees", "payroll_thousands",
               "emp_noise_flag"]]


def stage_cbp(engine, cbp):
    """Replace the rows of the keyed CBP table, then read them back."""
    table = f"{SCHEMA}.{CBP_TABLE}"
    with engine.begin() as conn:
        # TRUNCATE + append keeps the table's primary and foreign keys;
        # if_exists="replace" would drop and recreate it without them.
        conn.execute(text(f"TRUNCATE {table}"))
        cbp.to_sql(CBP_TABLE, conn, schema=SCHEMA, if_exists="append",
                   index=False)
    return pd.read_sql(f"SELECT * FROM {table}", engine)


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
        df = df.replace(SENTINELS, float("nan"))
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
    """Diagnose gaps, keep every county, flag missing store data, join CBP."""
    print("\nMissing share per column (Atlas):")
    print(df.isna().mean().sort_values(ascending=False).round(3))

    # Only counties missing the access measures themselves are dropped;
    # imputing them would invent the quantity being ranked.
    before = len(df)
    df = df.dropna(subset=CORE_ACCESS).copy()
    print(f"Dropped {before - len(df)} counties missing access measures")

    # Store counts are NOT dropped or imputed. Dropping the ~half of
    # counties coded -9999 (mostly rural) skewed the first ranking toward
    # large urban counties, a selection bias. They stay in, flagged, and
    # are scored on the measures they have.
    df["store_data_missing"] = df["grocpth"].isna()
    print(f"Counties with no store count (kept, flagged): "
          f"{df['store_data_missing'].sum()}")

    # LEFT JOIN keeps counties with no grocery record in Census.
    df = pd.merge(df, cbp, on="fips", how="left")
    df["cbp_status"] = df["establishments"].notna().map(
        {True: "reported", False: "no_cbp_record"})
    # A missing CBP row is NOT assumed to be zero; counts stay NaN.
    return df


def weighted_score(ranks, weights):
    """Weighted mean of percentile ranks over the measures a county has.

    Missing measures are left out of both the numerator and the
    denominator, so a county is not penalized for a data gap.
    """
    w = pd.Series(weights)[ranks.columns]
    total = ranks.mul(w).sum(axis=1, min_count=1)
    used = ranks.notna().mul(w).sum(axis=1)
    return total / used


def engineer_features(df):
    """Add population, density, per-store metrics and priority scores."""
    df = df.copy()
    # Estimated population = low-access residents / low-access share.
    share = df["pct_laccess_pop"].where(df["pct_laccess_pop"] > 0) / 100
    df["population"] = (df["laccess_pop"] / share).round()
    # Lower density of stores = greater need, so flip the sign.
    df["low_store_density"] = -df["grocpth"]
    df["lowi_per_store"] = df["laccess_lowi"] / (df["groc"] + 1)
    df["payroll_per_estab"] = (df["payroll_thousands"]
                               / df["establishments"])

    ranks = df[NEED_METRICS].rank(pct=True)
    df["n_measures"] = ranks.notna().sum(axis=1)
    df["priority_score"] = weighted_score(ranks, BASE_WEIGHTS)
    df["priority_rank"] = df["priority_score"].rank(
        ascending=False, method="first").astype(int)
    # Sensitivity check: triple weight on low-income measures.
    df["alt_score"] = weighted_score(ranks, ALT_WEIGHTS)
    df["alt_rank"] = df["alt_score"].rank(
        ascending=False, method="first").astype(int)
    return df.sort_values("priority_rank")


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def diagnostic_plots(df):
    """Histograms and a boxplot to check distributions and outliers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["grocpth"].dropna(), bins=30, ax=axes[0])
    axes[0].set_title("Grocery stores per 1,000 residents")
    sns.histplot(df["pct_laccess_lowi"], bins=30, ax=axes[1])
    axes[1].set_title("Low-income, low-access share (%)")
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

    plotted = df.dropna(subset=["grocpth"])  # counties with a store count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plotted, x="grocpth", y="pct_laccess_lowi",
                    hue=plotted["priority_rank"] <= 10, alpha=0.7, ax=ax)
    ax.set_title("Store availability vs. low-income, low-access share "
                 f"({len(plotted)} counties with a store count)")
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
    cbp = stage_cbp(engine, clean_cbp(fetch_cbp()))

    df = engineer_features(handle_missing(atlas, cbp))
    diagnostic_plots(df)
    stakeholder_charts(df)

    cols = ["priority_rank", "county", "population", "laccess_pop",
            "pct_laccess_pop", "laccess_lowi", "pct_laccess_lowi",
            "grocpth", "establishments", "cbp_status", "n_measures",
            "priority_score", "alt_rank"]
    top10 = df.head(10)
    top10[cols].to_csv(OUTPUT_DIR / "top10_table.csv", index=False)
    df[cols].to_csv(OUTPUT_DIR / "all_counties_ranked.csv", index=False)
    overlap = len(set(top10["fips"])
                  & set(df.nsmallest(10, "alt_rank")["fips"]))
    print(top10[cols].to_string(index=False))
    print(f"\nTop-10 overlap under alternative weights: {overlap}/10")
    print(df["cbp_status"].value_counts())


if __name__ == "__main__":
    main()
