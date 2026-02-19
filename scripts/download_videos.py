"""
Download videos for multimodal SAE training.

Usage:
    python scripts/download_videos.py celebdf --output-dir ./videos --max-videos 10
    python scripts/download_videos.py action100m --output-dir ./videos --max-videos 1000
"""

import logging
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ── CLI entrypoint ──────────────────────────────────────────────────────────


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Download videos for multimodal SAE training."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


# ── Subcommands ─────────────────────────────────────────────────────────────


@cli.command()
@click.option("-o", "--output-dir", type=click.Path(), default="./videos", show_default=True)
@click.option("-n", "--max-videos", type=int, default=None, help="Max videos to keep (default: all)")
def celebdf(output_dir: str, max_videos: int | None):
    """Download CelebDFv2 deepfake dataset from Kaggle."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _load_kaggle_credentials()

    temp_dir = output / "temp_kaggle_download"
    temp_dir.mkdir(exist_ok=True)

    zip_path = _kaggle_download("reubensuju/celeb-df-v2", temp_dir)

    extract_dir = temp_dir / "extracted"
    _extract_zip(zip_path, extract_dir)

    videos = _find_videos(extract_dir)
    logger.info(f"Found {len(videos)} videos in dataset")

    if max_videos:
        videos = videos[:max_videos]

    for video_path in tqdm(videos, desc="Copying videos"):
        relative = video_path.relative_to(extract_dir)
        dest = output / str(relative).replace(os.sep, "_")
        shutil.copy2(video_path, dest)

    click.echo(f"Done: {len(videos)} videos copied to {output}")


@cli.command()
@click.option("-o", "--output-dir", type=click.Path(), default="./videos", show_default=True)
@click.option("-n", "--max-videos", type=int, default=10000, show_default=True)
@click.option("--max-duration", type=int, default=300, show_default=True, help="Skip videos longer than N seconds")
@click.option("--dataset-path", default="facebook/Action100M-preview", show_default=True)
@click.option("-j", "--workers", type=int, default=4, show_default=True, help="Parallel download workers")
def action100m(output_dir: str, max_videos: int, max_duration: int, dataset_path: str, workers: int):
    """Download Action100M videos from YouTube via yt-dlp."""
    from datasets import load_dataset

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{dataset_path}/data/*.parquet",
        streaming=True,
        split="train",
    )

    # Collect unique video UIDs
    seen = set()
    uids = []
    for row in dataset:
        uid = row.get("video_uid")
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
        if len(uids) >= max_videos:
            break

    click.echo(f"Downloading {len(uids)} unique videos to {output}")

    success = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_youtube_video, uid, output, max_duration): uid for uid in uids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if future.result():
                success += 1

    click.echo(f"Done: {success}/{len(uids)} succeeded")


# ── Helpers (celebdf) ───────────────────────────────────────────────────────


def _load_kaggle_credentials():
    """Load Kaggle credentials from /workspace/secrets/kaggle_api if available."""
    secrets = Path("/workspace/secrets/kaggle_api")
    if not secrets.exists():
        return
    for line in secrets.read_text().splitlines():
        if line.startswith("export "):
            key, value = line[7:].strip().split("=", 1)
            os.environ[key] = value


def _kaggle_download(dataset: str, dest: Path) -> Path:
    """Download a Kaggle dataset zip to dest/. Returns zip path."""
    zip_path = dest / f"{dataset.replace('/', '_')}.zip"
    if zip_path.exists():
        logger.info(f"Using cached download: {zip_path}")
        return zip_path

    logger.info(f"Downloading {dataset} from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest)],
        check=True, capture_output=True, text=True,
    )
    downloaded = next(dest.glob("*.zip"))
    downloaded.rename(zip_path)
    return zip_path


def _extract_zip(zip_path: Path, extract_dir: Path):
    """Extract a zip file, falling back to system unzip on failure."""
    if extract_dir.exists():
        logger.info(f"Using cached extraction: {extract_dir}")
        return

    logger.info(f"Extracting {zip_path.stat().st_size / 1e9:.2f} GB...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"Python zipfile failed ({e}), trying system unzip...")
        subprocess.run(
            ["unzip", "-q", str(zip_path), "-d", str(extract_dir)],
            check=True, capture_output=True, text=True,
        )


def _find_videos(directory: Path) -> list[Path]:
    """Recursively find all video files in a directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.rglob(f"*{ext}"))
    return videos


# ── Helpers (action100m) ────────────────────────────────────────────────────


def _download_youtube_video(video_uid: str, output_dir: Path, max_duration: int) -> bool:
    """Download a single YouTube video by ID. Returns True on success."""
    output_path = output_dir / f"{video_uid}.mp4"
    if output_path.exists():
        return True

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
        "--merge-output-format", "mp4",
        "--match-filter", f"duration<{max_duration}",
        "--no-playlist", "--quiet",
        "--output", str(output_path),
        f"https://www.youtube.com/watch?v={video_uid}",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


if __name__ == "__main__":
    cli()
