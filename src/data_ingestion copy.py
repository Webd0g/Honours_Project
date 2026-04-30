import requests as re
import pandas as pd
from pathlib import Path

api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"

OUTPUT_PATH = Path("extragalactic_candidates.parquet")
CHUNK_SIZE = 1000
TARGET = 10000

def fetch_tag_chunk(tag: str, startdate: str, stopdate: str, n: int) -> list:
    r = re.post(
        api_url_tags,
        json={
            "tag": tag,
            "n": n,
            "startdate": startdate,
            "stopdate": stopdate,
            "columns": "r:diaObjectId",
        },
    )
    r.raise_for_status()
    return r.json()

def build_candidate_catalogue(
    tag: str,
    startdate: str,
    stopdate: str,
    target: int = TARGET,
    chunk_size: int = CHUNK_SIZE,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    
    seen_ids = set()
    records = []

    for i in range(0, target, chunk_size):
        print(f"Fetching chunk {i // chunk_size + 1}...")
        chunk = fetch_tag_chunk(tag, startdate, stopdate, n=chunk_size)

        for val in chunk:
            oid = str(val["r:diaObjectId"])
            if oid not in seen_ids:
                seen_ids.add(oid)
                records.append({"diaObjectId": oid})

        print(f"  Unique candidates so far: {len(seen_ids)}")

        # Early exit if the API is returning fewer results than requested
        if len(chunk) < chunk_size:
            print("API returned fewer results than requested, stopping early.")
            break

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} unique candidates to {output_path}")
    return df


if __name__ == "__main__":
    df = build_candidate_catalogue(
        tag="extragalactic_lt20mag_candidate",
        startdate="2026-02-19",
        stopdate="2026-02-27",
    )