import pathlib

import pytest

from bio2zarr import vcf2zarr
from vcztools.vcf_writer import write_vcf
from .utils import assert_vcfs_close

import shutil

import pysam
from hypothesis import HealthCheck, Phase, given, note, settings
from hypothesis_vcf import vcf

from bio2zarr import vcf2zarr


def vcz_path_cache(vcf_path):
    """
    Store converted files in a cache to speed up tests. We're not testing
    vcf2zarr here, so no point in running over and over again.
    """
    cache_path = pathlib.Path("vcz_test_cache")
    if not cache_path.exists():
        cache_path.mkdir()
    cached_vcz_path = (cache_path / vcf_path.name).with_suffix(".vcz")
    if not cached_vcz_path.exists():
        vcf2zarr.convert([vcf_path], cached_vcz_path, worker_processes=0, local_alleles=False)
    return cached_vcz_path


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "1kg_2020_chr20_annotations.bcf",
        "1kg_2020_chrM.vcf.gz",
        "field_type_combos.vcf.gz",
    ],
)
@pytest.mark.parametrize("implementation", ["c"])  # , "numba"])
def test_vcf_to_zarr_to_vcf__real_files(tmp_path, vcf_file, implementation):
    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = vcz_path_cache(original)
    generated = tmp_path.joinpath("output.vcf")
    write_vcf(vcz, generated, implementation=implementation)
    assert_vcfs_close(original, generated)


# Make sure POS starts at 1, since CSI indexing doesn't seem to support zero-based
# coordinates (even when passing zerobased=True to pysam.tabix_index below)
@given(vcf_string=vcf(min_pos=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], max_examples=10)
def test_vcf_to_zarr_to_vcf__hypothesis_generated_vcf(tmp_path, vcf_string):
    note(f"vcf:\n{vcf_string}")

    path = tmp_path / "input.vcf"
    icf_path = tmp_path / "icf"
    zarr_path = tmp_path / "zarr"
    output = tmp_path / "output.vcf"

    with open(path, "w") as f:
        f.write(vcf_string)

    # make sure outputs don't exist (from previous hypothesis example)
    shutil.rmtree(str(icf_path), ignore_errors=True)
    shutil.rmtree(str(zarr_path), ignore_errors=True)

    # create a tabix index for the VCF,
    # using CSI since POS can exceed range supported by TBI
    # (this also compresses the input file)
    pysam.tabix_index(str(path), preset="vcf", force=True, csi=True)

    # test that we can convert VCFs to Zarr and back unchanged
    vcf2zarr.convert(
        [str(path) + ".gz"], zarr_path, icf_path=icf_path, worker_processes=0
    )
    write_vcf(zarr_path, output, implementation="c")
    assert_vcfs_close(str(path) + ".gz", output)
