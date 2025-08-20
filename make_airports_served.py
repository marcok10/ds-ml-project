# make_airports_served.py
import pandas as pd
from pathlib import Path

ROUTE_FILE = Path("artifacts/route_expected_duration.csv")
OUT_FILE   = Path("artifacts/airports_served.csv")

def main():
    df = pd.read_csv(ROUTE_FILE)
    # route column looks like "AAA-BBB"
    codes = set()
    for r in df["route"].astype(str):
        if "-" in r:
            a, b = r.split("-", 1)
            codes.add(a.strip())
            codes.add(b.strip())

    # simple CSV with just the IATA codes you actually fly
    out = pd.DataFrame(sorted(codes), columns=["iata"])
    OUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE} with {len(out)} airports")

if __name__ == "__main__":
    main()
