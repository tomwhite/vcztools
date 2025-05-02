import xarray as xr

from vcztools import filter as filter_mod
from vcztools.regions import (
    parse_regions,
    parse_targets,
    regions_to_chunk_indexes,
    regions_to_selection,
)
from vcztools.samples import parse_samples


# TODO: would be nice not to need root
def filter_regions(ds, root, regions=None, targets=None):
    if regions is not None or targets is not None:
        variant_selection = regions_to_variant_selection(
            root, variant_regions=regions, variant_targets=targets
        )
        ds = ds.isel(variants=variant_selection)
    return ds


def filter_expressions(ds, include=None, exclude=None):
    filter_expr = filter_mod.FilterExpression(
        field_names=set(ds.data_vars), include=include, exclude=exclude
    )

    def compute_call_mask(ds):
        call_mask = filter_expr.evaluate(ds)
        return xr.DataArray(call_mask, dims=["variants", "samples"])

    # restrict to fields needed by filter expression
    ds_filter_fields = ds[list(filter_expr.referenced_fields)]
    # note that this will only work if chunked in the variants dimension
    # may need to merge chunks in samples dim
    da = xr.map_blocks(compute_call_mask, ds_filter_fields)
    ds["call_mask"] = da

    # filter to variants where at least one sample has been selected
    ds = ds.isel(variants=ds.call_mask.any(dim="samples"))
    return ds


def filter_samples(ds, samples=None):
    if samples is not None:
        all_samples = ds["sample_id"].values
        _, sample_selection = parse_samples(samples, all_samples)
        ds = ds.isel(samples=sample_selection)

    return ds


# TODO: this was copied from vcf_writer - needs to be a utility
# and maybe use xarray? (although not sure how to do block selection efficiently)
# map_blocks with an arg indicating which blocks to use? apply_ufunc?
def regions_to_variant_selection(root, variant_regions=None, variant_targets=None):
    contigs_u = root["contig_id"][:].astype("U").tolist()
    regions = parse_regions(variant_regions, contigs_u)
    targets, complement = parse_targets(variant_targets, contigs_u)

    # Use the region index to find the chunks that overlap specfied regions or
    # targets
    region_index = root["region_index"][:]
    chunk_indexes = regions_to_chunk_indexes(
        regions,
        targets,
        complement,
        region_index,
    )

    # Then use only load required variant_contig/position chunks
    if len(chunk_indexes) == 0:
        # no chunks - no variants to write
        return
    elif len(chunk_indexes) == 1:
        # single chunk
        block_sel = chunk_indexes[0]
    else:
        # zarr.blocks doesn't support int array indexing - use that when it does
        block_sel = slice(chunk_indexes[0], chunk_indexes[-1] + 1)

    region_variant_contig = root["variant_contig"].blocks[block_sel][:]
    region_variant_position = root["variant_position"].blocks[block_sel][:]
    region_variant_length = root["variant_length"].blocks[block_sel][:]

    # Find the final variant selection
    return regions_to_selection(
        regions,
        targets,
        complement,
        region_variant_contig,
        region_variant_position,
        region_variant_length,
    )
