
import csv
import os
import time
import requests

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("CENSUS_API_KEY")

if not API_KEY or API_KEY == "YOUR_ACTUAL_KEY":
    raise ValueError(
        "Census API key not found. Check your .env file."
    )

API_URL = "https://api.census.gov/data/2023/cbp"

OUTPUT_FILE = "data/cbp_2023_county_business_patterns.csv"

# Texas only
STATE_FIPS = ["48"]

# All-industry total and all available 2-digit NAICS sectors
INDUSTRY_CODES = [
    "00",
    "11", "21", "22", "23",
    "31", "32", "33",
    "42", "44", "45", "48", "49",
    "51", "52", "53", "54", "55", "56",
    "61", "62", "71", "72", "81", "92"
]

# Census variables
GET_FIELDS = (
    "NAME,NAICS2017_LABEL,"
    "ESTAB,EMP,PAYQTR1,PAYANN"
)


# --------------------------------------------------
# 2. CREATE API SESSION WITH AUTOMATIC RETRIES
# --------------------------------------------------

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)


# --------------------------------------------------
# 3. PREPARE OUTPUT FILE
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

total_records = 0
successful_requests = 0
empty_requests = 0

total_requests = len(STATE_FIPS) * len(INDUSTRY_CODES)

print("Starting 2023 Texas Census CBP download...")
print(f"Geographies: {len(STATE_FIPS)}")
print(f"Industry categories: {len(INDUSTRY_CODES)}")
print(f"Planned API requests: {total_requests}")
print()


# --------------------------------------------------
# 4. DOWNLOAD DATA FROM CENSUS API
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = None
    expected_header = None

    for state_fips in STATE_FIPS:

        state_records = 0

        for industry in INDUSTRY_CODES:

            # 00 = all-industry total
            # Other codes retrieve the industry hierarchy
            if industry == "00":
                industry_filter = "00"
            else:
                industry_filter = industry + "*"

            params = {
                "get": GET_FIELDS,
                "for": "county:*",
                "in": f"state:{state_fips}",
                "NAICS2017": industry_filter,
                "LFO": "001",
                "EMPSZES": "001",
                "key": API_KEY
            }

            print(
                f"Requesting Texas, industry {industry_filter}...",
                end=" ",
                flush=True
            )

            response = session.get(
                API_URL,
                params=params,
                timeout=90
            )

            # HTTP 204 means no data is available
            if response.status_code == 204:
                empty_requests += 1
                print("No records available (HTTP 204).")
                continue

            # Check other HTTP errors
            if not response.ok:
                raise RuntimeError(
                    f"\nCensus API HTTP error.\n"
                    f"Industry: {industry_filter}\n"
                    f"Status: {response.status_code}\n"
                    f"Response: {response.text[:1000]}"
                )

            # Convert API response to JSON
            try:
                results = response.json()

            except requests.exceptions.JSONDecodeError:
                raise RuntimeError(
                    f"\nCensus API returned a non-JSON response.\n"
                    f"Industry: {industry_filter}\n"
                    f"HTTP status: {response.status_code}\n"
                    f"Response: {response.text[:1000]}"
                )

            if not isinstance(results, list) or len(results) <= 1:
                empty_requests += 1
                print("No records returned.")
                continue

            header = results[0]

            # Create CSV header using the first response
            if writer is None:

                expected_header = header + [
                    "county_fips",
                    "data_year"
                ]

                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=expected_header
                )

                writer.writeheader()

            # Confirm consistent columns
            elif header != expected_header[:-2]:

                raise RuntimeError(
                    f"Unexpected column structure "
                    f"for industry {industry_filter}."
                )

            # Write county records to CSV
            for values in results[1:]:

                record = dict(zip(header, values))

                state_code = str(record["state"]).zfill(2)
                county_code = str(record["county"]).zfill(3)

                record["state"] = state_code
                record["county"] = county_code

                record["county_fips"] = (
                    state_code + county_code
                )

                record["data_year"] = "2023"

                writer.writerow(record)

                total_records += 1
                state_records += 1

            successful_requests += 1

            print(f"{len(results) - 1:,} records")

            # Save progress after every successful request
            csvfile.flush()

            time.sleep(0.2)

        print(
            f"\nTexas download progress: "
            f"{state_records:,} records\n"
        )


# --------------------------------------------------
# 5. FINAL DOWNLOAD SUMMARY
# --------------------------------------------------

print("=" * 45)
print("TEXAS CENSUS DOWNLOAD COMPLETED")
print("=" * 45)

print(f"Total records: {total_records:,}")
print(f"Successful requests: {successful_requests:,}")
print(f"Empty requests: {empty_requests:,}")
print(f"Output file: {OUTPUT_FILE}")

print("=" * 45)








