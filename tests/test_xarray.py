import pathlib

import pytest
import xarray as xr
import zarr

from vcztools.xarray import filter_expressions, filter_regions, filter_samples

from .utils import vcz_path_cache


@pytest.fixture()
def vcz():
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    return vcz_path_cache(original)


def test_regions(vcz):
    ds = xr.open_zarr(vcz, concat_characters=False)
    root = zarr.open(vcz, mode="r")
    ds = filter_regions(ds, root, regions="20:1230236-")
    print_dataset(ds)


def test_samples(vcz):
    ds = xr.open_zarr(vcz, concat_characters=False)

    ds = filter_samples(ds, samples="NA00002,NA00003")
    print_dataset(ds)


def test_filter_expressions(vcz):
    ds = xr.open_zarr(vcz, concat_characters=False)

    ds = filter_expressions(ds, include="FMT/DP>3")
    print_dataset(ds)


def test_all_filters(vcz):
    ds = xr.open_zarr(vcz, concat_characters=False)

    root = zarr.open(vcz, mode="r")
    ds = filter_regions(ds, root, regions="20:1230236-")
    ds = filter_expressions(ds, include="FMT/DP>3")
    ds = filter_samples(ds, samples="NA00002,NA00003")
    print_dataset(ds)


def print_dataset(ds):
    if "call_mask" in ds:
        for contig, pos, dp, mask in zip(
            ds.variant_contig.values,
            ds.variant_position.values,
            ds.call_DP.values,
            ds.call_mask.values,
        ):
            print(f"{contig} {pos} DP={dp} mask={mask}")
    else:
        for contig, pos, dp in zip(
            ds.variant_contig.values, ds.variant_position.values, ds.call_DP.values
        ):
            print(f"{contig} {pos} DP={dp}")
