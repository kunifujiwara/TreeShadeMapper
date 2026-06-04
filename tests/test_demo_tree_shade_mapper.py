"""Tests derived from demo/Demo_tree_shade_mapper.ipynb."""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests

from tree_shade_mapper import get_tree_shade


DEMO_ZIP_URL = "https://github.com/kunifujiwara/data/blob/master/tree_shade_mapper/data.zip"
DEMO_TIME_START = "2024-01-01 07:00:00"
DEMO_TIME_END = "2024-01-01 20:00:00"
DEMO_INTERVAL = "60min"
DEMO_TIME_ZONE = "Asia/Singapore"
DEMO_LATITUDE = 1.29751
DEMO_LONGITUDE = 103.77012
DEMO_IMAGE_SIZE = (512, 256)
DEMO_CALC_TYPE = "map"
DEMO_VMIN = 0
DEMO_VMAX = 1200
DEMO_RESOLUTION = 14


def github_blob_url_to_raw_url(url: str) -> str:
    """Convert a GitHub blob URL to its raw content URL."""
    return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")


def download_and_extract_zip(url: str, extract_path: Path) -> None:
    """Download the demo zip archive and extract it into extract_path."""
    raw_url = github_blob_url_to_raw_url(url)
    response = requests.get(raw_url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(extract_path)


def test_demo_imports_get_tree_shade() -> None:
    assert callable(get_tree_shade)


def test_demo_uses_expected_parameters() -> None:
    assert DEMO_CALC_TYPE == "map"
    assert DEMO_TIME_ZONE == "Asia/Singapore"
    assert DEMO_IMAGE_SIZE == (512, 256)
    assert DEMO_LATITUDE == pytest.approx(1.29751)
    assert DEMO_LONGITUDE == pytest.approx(103.77012)


@pytest.mark.skipif(
    os.getenv("RUN_DEMO_TREE_SHADE_MAPPER") != "1",
    reason="Set RUN_DEMO_TREE_SHADE_MAPPER=1 to download data and run the full demo.",
)
def test_demo_tree_shade_mapper_end_to_end(tmp_path: Path) -> None:
    extract_path = tmp_path / "extracted_data"
    extract_path.mkdir()
    download_and_extract_zip(DEMO_ZIP_URL, extract_path)

    base_dir = extract_path / "data"
    get_tree_shade(
        str(base_dir),
        DEMO_TIME_START,
        DEMO_TIME_END,
        DEMO_INTERVAL,
        DEMO_TIME_ZONE,
        DEMO_LATITUDE,
        DEMO_LONGITUDE,
        image_size=DEMO_IMAGE_SIZE,
        calc_type=DEMO_CALC_TYPE,
        vmin=DEMO_VMIN,
        vmax=DEMO_VMAX,
        resolution=DEMO_RESOLUTION,
    )

    assert (base_dir / "frames_svf.csv").exists()
    assert (base_dir / "frames_solar.csv").exists()
    assert (base_dir / "frames_solar_accu.csv").exists()

    binary_arrays = sorted((base_dir / "binary_tcm").glob("*_bin.npy"))
    assert binary_arrays
    assert any(np.load(binary_path).mean() > 0 for binary_path in binary_arrays)
