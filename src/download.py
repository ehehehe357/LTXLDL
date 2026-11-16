"""Tải 4 mốc Inside Airbnb của thành phố trong config, rồi lọc theo ID của nhóm."""
import argparse
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE = "https://data.insideairbnb.com"
FILES = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]
RAW, OUT, IDS = Path("data/raw"), Path("data/processed"), Path("samples")


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print("  đã có:", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("  tải:", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def filter_listings(path, ids):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["id"].isin(ids["listing_id"])]
    return df.merge(ids.rename(columns={"listing_id": "id"}), on="id", how="left")


def filter_calendar(path, listing_ids):
    parts = []
    for chunk in pd.read_csv(path, usecols=["listing_id", "date", "available"],
                             chunksize=500_000):
        parts.append(chunk[chunk["listing_id"].isin(listing_ids)])
    return pd.concat(parts, ignore_index=True)


def filter_reviews(path, rev):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["id"].isin(rev["review_id"])]
    flags = rev.rename(columns={"review_id": "id"})[["id", "llm_sample", "hand_label"]]
    return df.merge(flags, on="id", how="left")


def check(l, c, r, is_last):
    print(f"  dòng: listings={len(l)}, calendar={len(c)}, reviews={len(r)}")
    print("  room_type:", l["room_type"].value_counts().to_dict())
    print("  số listing number_of_reviews=0:", int((l["number_of_reviews"] == 0).sum()))
    if is_last:
        print("  số listing thiếu price (mốc cuối, phải = 20):", int(l["price"].isna().sum()))
    print("  số listing mỗi vùng:", l["analysis_area"].value_counts().sort_index().to_dict())


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    lid = pd.read_csv(IDS / "listing_ids.csv")
    rid = pd.read_csv(IDS / "review_ids.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    snaps = [str(s) for s in cfg["snapshots"]]
    for i, snap in enumerate(snaps):
        print(f"== Mốc {snap}")
        for name in FILES:
            url = f"{BASE}/{cfg['data_root']}/{snap}/data/{name}"
            download(url, RAW / snap / name)

        l = filter_listings(RAW / snap / "listings.csv.gz", lid)
        c = filter_calendar(RAW / snap / "calendar.csv.gz", set(lid["listing_id"]))
        r = filter_reviews(RAW / snap / "reviews.csv.gz", rid)
        check(l, c, r, is_last=(i == len(snaps) - 1))

        l.to_csv(OUT / f"listings_{snap}.csv.gz", index=False)
        c.to_csv(OUT / f"calendar_{snap}.csv.gz", index=False)
        r.to_csv(OUT / f"reviews_{snap}.csv.gz", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)