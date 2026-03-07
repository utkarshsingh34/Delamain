"""Download NYC TLC trip data and taxi zone files.

Downloads yellow taxi Parquet for January 2023 and June 2023,
taxi zone shapefile, and zone lookup CSV to data/raw/.
"""

import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"

DOWNLOADS: list[tuple[str, str]] = [
    (f"{BASE_URL}/trip-data/yellow_tripdata_2023-01.parquet", "yellow_tripdata_2023-01.parquet"),
    (f"{BASE_URL}/trip-data/yellow_tripdata_2023-06.parquet", "yellow_tripdata_2023-06.parquet"),
    (f"{BASE_URL}/misc/taxi_zones.zip", "taxi_zones.zip"),
    (f"{BASE_URL}/misc/taxi_zone_lookup.csv", "taxi_zone_lookup.csv"),
]


def get_raw_dir() -> Path:
    """Return the data/raw directory, creating it if needed."""
    raw_dir = Path(__file__).resolve().parents[1] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_file(url: str, dest: Path) -> None:
    """Download a file with a tqdm progress bar. Skips if dest already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Skipping {dest.name} (already exists, {dest.stat().st_size:,} bytes)")
        return

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as progress:
            for chunk in response.iter_bytes(chunk_size=65536):
                f.write(chunk)
                progress.update(len(chunk))


def extract_shapefile(zip_path: Path, extract_dir: Path) -> None:
    """Extract taxi zones shapefile zip into extract_dir."""
    if extract_dir.exists() and any(extract_dir.glob("*.shp")):
        print(f"  Skipping extraction (already extracted to {extract_dir.name}/)")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    shp_count = len(list(extract_dir.rglob("*.shp")))
    print(f"  Extracted {len(zf.namelist())} files to {extract_dir.name}/ ({shp_count} .shp)")


def print_summary(raw_dir: Path) -> None:
    """Print summary of downloaded files and sizes."""
    print("\n--- Download Summary ---")
    total_bytes = 0
    for _, filename in DOWNLOADS:
        path = raw_dir / filename
        if path.exists():
            size = path.stat().st_size
            total_bytes += size
            print(f"  {filename:<45} {size:>12,} bytes  ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {filename:<45} MISSING")

    zones_dir = raw_dir / "taxi_zones"
    if zones_dir.exists():
        zone_files = list(zones_dir.rglob("*"))
        zone_size = sum(f.stat().st_size for f in zone_files if f.is_file())
        total_bytes += zone_size
        print(f"  {'taxi_zones/ (extracted)':<45} {zone_size:>12,} bytes  ({zone_size / 1024 / 1024:.1f} MB)")

    print(f"\n  Total: {total_bytes / 1024 / 1024:.1f} MB")


def main() -> None:
    """Download all required data files."""
    raw_dir = get_raw_dir()
    print(f"Downloading to: {raw_dir}\n")

    for url, filename in DOWNLOADS:
        dest = raw_dir / filename
        download_file(url, dest)

    # Extract shapefile
    zip_path = raw_dir / "taxi_zones.zip"
    if zip_path.exists():
        extract_shapefile(zip_path, raw_dir / "taxi_zones")

    # Verify parquet files are readable
    import pyarrow.parquet as pq

    for _, filename in DOWNLOADS:
        if filename.endswith(".parquet"):
            path = raw_dir / filename
            table = pq.read_metadata(path)
            print(f"  Verified {filename}: {table.num_rows:,} rows, {table.num_columns} columns")

    print_summary(raw_dir)


if __name__ == "__main__":
    main()
