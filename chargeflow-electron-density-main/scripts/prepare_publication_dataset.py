#!/usr/bin/env python3
"""Prepare the ChargeFlow publication dataset bundle.

This script reuses verified prebuilt archives when they already match the paper,
builds the missing external-test tarballs directly from the paper list files, and
assembles a final publication package with manifests and metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path


FINAL_CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FINAL_CODE_ROOT.parent
DEFAULT_ARCHIVE_ROOT = Path("/vast/minhtrin/Charged_electron_density/Test-set-charge-electron-density")
DEFAULT_RELEASE_ROOT = Path("/vast/minhtrin/Charged_electron_density/dataset/chargeflow_jctc_release")
DEFAULT_OUTPUT_ROOT = Path("/vast/minhtrin/Charged_electron_density/dataset/chargeflow_publication_dataset")


@dataclass(frozen=True)
class SubsetSpec:
    subset_name: str
    archive_base: str
    list_d: Path
    list_l: Path
    list_dgs: Path
    list_lgs: Path
    expected_count: int


@dataclass(frozen=True)
class ArchivePresentation:
    source_name: str
    group_dir: str
    output_name: str
    paper_group: str
    subset_name: str
    density_role: str
    sample_count: int | None
    note: str


MISSING_SUBSET_SPECS = [
    SubsetSpec(
        subset_name="mofs",
        archive_base="mofs_rho",
        list_d=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_d_mofs",
        list_l=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_l_mofs",
        list_dgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_dgs_mofs",
        list_lgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_lgs_mofs",
        expected_count=120,
    ),
    SubsetSpec(
        subset_name="extreme_mofs",
        archive_base="extreme_mofs_rho",
        list_d=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_d_extreme_mofs",
        list_l=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_l_extreme_mofs",
        list_dgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_dgs_extreme_mofs",
        list_lgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_lgs_extreme_mofs",
        expected_count=123,
    ),
    SubsetSpec(
        subset_name="organic",
        archive_base="organic_rho",
        list_d=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_d_organic",
        list_l=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_l_organic",
        list_dgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_dgs_organic",
        list_lgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_lgs_organic",
        expected_count=12,
    ),
    SubsetSpec(
        subset_name="extreme_organic",
        archive_base="extreme_organic_rho",
        list_d=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_d_extreme_organic",
        list_l=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_l_extreme_organic",
        list_dgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_dgs_extreme_organic",
        list_lgs=FINAL_CODE_ROOT / "data_lists" / "test_sets" / "list_lgs_extreme_organic",
        expected_count=12,
    ),
]


ARCHIVE_PRESENTATIONS = [
    ArchivePresentation(
        source_name="MP_neutral_rho_initial.tar.gz",
        group_dir="training_source_corpora",
        output_name="materials_project_source_inputs_neutral_initial_density.tar.gz",
        paper_group="training_source_corpora",
        subset_name="materials_project",
        density_role="input_initial_density",
        sample_count=None,
        note="Source corpus for paper train and internal-development-holdout inputs; exact paper samples are defined by manifests.",
    ),
    ArchivePresentation(
        source_name="subMP12k_charged_rho.tar.gz",
        group_dir="training_source_corpora",
        output_name="materials_project_source_targets_charged_dft_density.tar.gz",
        paper_group="training_source_corpora",
        subset_name="materials_project",
        density_role="target_dft_density",
        sample_count=None,
        note="Source corpus for paper train and internal-development-holdout targets; exact paper samples are defined by manifests.",
    ),
    ArchivePresentation(
        source_name="perovskites_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="perovskites_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="perovskites",
        density_role="target_dft_density",
        sample_count=159,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="perovskites_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="perovskites_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="perovskites",
        density_role="input_initial_density",
        sample_count=159,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="defects_charged_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="charged_defects_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="charged_defects",
        density_role="target_dft_density",
        sample_count=1149,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="defects_charged_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="charged_defects_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="charged_defects",
        density_role="input_initial_density",
        sample_count=1149,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="MultisiteDefects_Diamond_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="multisite_diamond_defects_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="multisite_diamond_defects",
        density_role="target_dft_density",
        sample_count=54,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="MultisiteDefects_Diamond_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="multisite_diamond_defects_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="multisite_diamond_defects",
        density_role="input_initial_density",
        sample_count=54,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="SpecialDefects_Diamond_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="special_diamond_defects_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="special_diamond_defects",
        density_role="target_dft_density",
        sample_count=42,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="SpecialDefects_Diamond_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="special_diamond_defects_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="special_diamond_defects",
        density_role="input_initial_density",
        sample_count=42,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="mofs_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="mofs_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="mofs",
        density_role="target_dft_density",
        sample_count=120,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="mofs_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="mofs_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="mofs",
        density_role="input_initial_density",
        sample_count=120,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="extreme_mofs_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="extreme_mofs_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="extreme_mofs",
        density_role="target_dft_density",
        sample_count=123,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="extreme_mofs_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="extreme_mofs_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="extreme_mofs",
        density_role="input_initial_density",
        sample_count=123,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="organic_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="organic_crystals_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="organic_crystals",
        density_role="target_dft_density",
        sample_count=12,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="organic_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="organic_crystals_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="organic_crystals",
        density_role="input_initial_density",
        sample_count=12,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="extreme_organic_rho.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="extreme_organic_crystals_targets_dft_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="extreme_organic_crystals",
        density_role="target_dft_density",
        sample_count=12,
        note="External periodic benchmark subset.",
    ),
    ArchivePresentation(
        source_name="extreme_organic_rho_initial.tar.gz",
        group_dir="external_periodic_benchmark",
        output_name="extreme_organic_crystals_inputs_initial_density.tar.gz",
        paper_group="external_periodic_benchmark",
        subset_name="extreme_organic_crystals",
        density_role="input_initial_density",
        sample_count=12,
        note="External periodic benchmark subset.",
    ),
]


ARCHIVE_PRESENTATION_BY_SOURCE = {
    entry.source_name: entry for entry in ARCHIVE_PRESENTATIONS
}


MANIFEST_RENAMES = {
    "all_paper_splits.csv": "paper_splits_all.csv",
    "mp_charged_train.csv": "materials_project_train_split.csv",
    "mp_charged_internal_holdout.csv": "materials_project_internal_development_holdout_split.csv",
    "perovskites.csv": "external_periodic_test_perovskites_split.csv",
    "charged_defects.csv": "external_periodic_test_charged_defects_split.csv",
    "multisite_defects_diamond.csv": "external_periodic_test_multisite_diamond_defects_split.csv",
    "special_defects_diamond.csv": "external_periodic_test_special_diamond_defects_split.csv",
    "mofs.csv": "external_periodic_test_mofs_split.csv",
    "extreme_mofs.csv": "external_periodic_test_extreme_mofs_split.csv",
    "organic.csv": "external_periodic_test_organic_crystals_split.csv",
    "extreme_organic.csv": "external_periodic_test_extreme_organic_crystals_split.csv",
    "summary.json": "paper_split_summary.json",
}


ZENODO_RENAMES = {
    "zenodo_metadata.template.json": "zenodo_metadata_template.json",
}


MANIFEST_ARCHIVE_MAP = {
    "training_input": {
        "public_archive": "training_source_corpora/materials_project_source_inputs_neutral_initial_density.tar.gz",
        "member_root": "MP_neutral_rho_initial",
    },
    "training_target": {
        "public_archive": "training_source_corpora/materials_project_source_targets_charged_dft_density.tar.gz",
        "member_root": "subMP12k_charged_rho",
    },
    "perovskites_input": {
        "public_archive": "external_periodic_benchmark/perovskites_inputs_initial_density.tar.gz",
        "member_root": "perovskites_rho_initial",
    },
    "perovskites_target": {
        "public_archive": "external_periodic_benchmark/perovskites_targets_dft_density.tar.gz",
        "member_root": "perovskites_rho",
    },
    "charged_defects_input": {
        "public_archive": "external_periodic_benchmark/charged_defects_inputs_initial_density.tar.gz",
        "member_root": "defects_charged_rho_initial",
    },
    "charged_defects_target": {
        "public_archive": "external_periodic_benchmark/charged_defects_targets_dft_density.tar.gz",
        "member_root": "defects_charged_rho",
    },
    "multisite_defects_diamond_input": {
        "public_archive": "external_periodic_benchmark/multisite_diamond_defects_inputs_initial_density.tar.gz",
        "member_root": "MultisiteDefects_Diamond_rho_initial",
    },
    "multisite_defects_diamond_target": {
        "public_archive": "external_periodic_benchmark/multisite_diamond_defects_targets_dft_density.tar.gz",
        "member_root": "MultisiteDefects_Diamond_rho",
    },
    "special_defects_diamond_input": {
        "public_archive": "external_periodic_benchmark/special_diamond_defects_inputs_initial_density.tar.gz",
        "member_root": "SpecialDefects_Diamond_rho_initial",
    },
    "special_defects_diamond_target": {
        "public_archive": "external_periodic_benchmark/special_diamond_defects_targets_dft_density.tar.gz",
        "member_root": "SpecialDefects_Diamond_rho",
    },
    "mofs_input": {
        "public_archive": "external_periodic_benchmark/mofs_inputs_initial_density.tar.gz",
        "member_root": "mofs_rho_initial",
    },
    "mofs_target": {
        "public_archive": "external_periodic_benchmark/mofs_targets_dft_density.tar.gz",
        "member_root": "mofs_rho",
    },
    "extreme_mofs_input": {
        "public_archive": "external_periodic_benchmark/extreme_mofs_inputs_initial_density.tar.gz",
        "member_root": "extreme_mofs_rho_initial",
    },
    "extreme_mofs_target": {
        "public_archive": "external_periodic_benchmark/extreme_mofs_targets_dft_density.tar.gz",
        "member_root": "extreme_mofs_rho",
    },
    "organic_input": {
        "public_archive": "external_periodic_benchmark/organic_crystals_inputs_initial_density.tar.gz",
        "member_root": "organic_rho_initial",
    },
    "organic_target": {
        "public_archive": "external_periodic_benchmark/organic_crystals_targets_dft_density.tar.gz",
        "member_root": "organic_rho",
    },
    "extreme_organic_input": {
        "public_archive": "external_periodic_benchmark/extreme_organic_crystals_inputs_initial_density.tar.gz",
        "member_root": "extreme_organic_rho_initial",
    },
    "extreme_organic_target": {
        "public_archive": "external_periodic_benchmark/extreme_organic_crystals_targets_dft_density.tar.gz",
        "member_root": "extreme_organic_rho",
    },
}


def read_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_existing_file(source: Path, dest: Path, link_mode: str, overwrite: bool) -> None:
    ensure_parent(dest)
    if dest.exists():
        if not overwrite:
            return
        dest.unlink()
    if link_mode == "hardlink":
        os.link(source, dest)
    elif link_mode == "symlink":
        os.symlink(source, dest)
    else:
        shutil.copy2(source, dest)


def sample_dirs_from_lists(rho_list: Path, grid_list: Path) -> dict[str, tuple[Path, Path]]:
    rho_paths = read_lines(rho_list)
    grid_paths = read_lines(grid_list)
    if len(rho_paths) != len(grid_paths):
        raise ValueError(f"List length mismatch: {rho_list} vs {grid_list}")

    mapping: dict[str, tuple[Path, Path]] = {}
    for rho_str, grid_str in zip(rho_paths, grid_paths):
        rho_path = Path(rho_str)
        grid_path = Path(grid_str)
        sample_id = rho_path.parent.name
        if sample_id in mapping:
            continue
        if not rho_path.exists():
            raise FileNotFoundError(f"Missing rho file: {rho_path}")
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing grid file: {grid_path}")
        mapping[sample_id] = (rho_path, grid_path)
    return mapping


def add_sample_tree_to_tar(
    archive: tarfile.TarFile,
    archive_root_name: str,
    sample_id: str,
    rho_path: Path,
    grid_path: Path,
) -> None:
    archive.add(rho_path, arcname=f"{archive_root_name}/{sample_id}/rho_22.npy", recursive=False)
    archive.add(
        grid_path,
        arcname=f"{archive_root_name}/{sample_id}/grid_sizes_22.dat",
        recursive=False,
    )


def build_subset_archives(
    spec: SubsetSpec,
    archive_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    target_mapping = sample_dirs_from_lists(spec.list_l, spec.list_lgs)
    input_mapping = sample_dirs_from_lists(spec.list_d, spec.list_dgs)

    if len(target_mapping) != spec.expected_count:
        raise ValueError(
            f"{spec.subset_name} target sample count {len(target_mapping)} "
            f"does not match expected {spec.expected_count}"
        )
    if len(input_mapping) != spec.expected_count:
        raise ValueError(
            f"{spec.subset_name} input sample count {len(input_mapping)} "
            f"does not match expected {spec.expected_count}"
        )
    if set(target_mapping) != set(input_mapping):
        raise ValueError(f"{spec.subset_name} input/target sample sets do not match")

    target_archive = archive_root / f"{spec.archive_base}.tar.gz"
    input_archive = archive_root / f"{spec.archive_base}_initial.tar.gz"
    archive_root.mkdir(parents=True, exist_ok=True)

    for archive_path in [target_archive, input_archive]:
        if archive_path.exists() and overwrite:
            archive_path.unlink()

    if not target_archive.exists():
        with tarfile.open(target_archive, "w:gz") as tf:
            for sample_id in sorted(target_mapping):
                rho_path, grid_path = target_mapping[sample_id]
                add_sample_tree_to_tar(tf, spec.archive_base, sample_id, rho_path, grid_path)

    if not input_archive.exists():
        with tarfile.open(input_archive, "w:gz") as tf:
            archive_root_name = f"{spec.archive_base}_initial"
            for sample_id in sorted(input_mapping):
                rho_path, grid_path = input_mapping[sample_id]
                add_sample_tree_to_tar(tf, archive_root_name, sample_id, rho_path, grid_path)

    return {
        "subset_name": spec.subset_name,
        "target_archive": str(target_archive),
        "input_archive": str(input_archive),
        "sample_count": spec.expected_count,
    }


def copy_named_files(
    source_root: Path,
    dest_root: Path,
    rename_map: dict[str, str],
    overwrite: bool,
) -> list[str]:
    copied: list[str] = []
    dest_root.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in rename_map.items():
        source = source_root / src_name
        if not source.exists():
            raise FileNotFoundError(f"Expected file not found: {source}")
        dest = dest_root / dest_name
        ensure_parent(dest)
        if dest.exists():
            if not overwrite:
                copied.append(str(dest))
                continue
            dest.unlink()
        shutil.copy2(source, dest)
        copied.append(str(dest))
    return copied


def load_release_summary(release_root: Path) -> dict:
    summary_path = release_root / "manifests" / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def build_package_readme(
    summary: dict,
    generated_subsets: list[dict[str, object]],
    archive_entries: list[dict[str, object]],
) -> str:
    generated_names = ", ".join(item["subset_name"] for item in generated_subsets) or "none"
    train_files = [
        entry["output_name"]
        for entry in archive_entries
        if entry["paper_group"] == "training_source_corpora"
    ]
    external_files = [
        entry["output_name"]
        for entry in archive_entries
        if entry["paper_group"] == "external_periodic_benchmark"
    ]
    return f"""# ChargeFlow Publication Dataset

This package accompanies the manuscript:

`ChargeFlow: Flow-Matching Refinement of Charge-Conditioned Electron Densities`

## Contents

- `training_source_corpora/`
  - two source-corpus archives used to form the manuscript train and internal-development-holdout splits
- `external_periodic_benchmark/`
  - input/target archive pairs for all 8 external benchmark subsets
- `manifests/`
  - exact paper split manifests with paper-facing names and no local filesystem paths
- `archive_inventory.csv`
  - one-line explanation of every archive in the package
- `zenodo/`
  - Zenodo metadata template
- `package_summary.json`
  - machine-readable packaging summary

## Paper Split Counts

- train: {summary.get('splits', {}).get('train', {}).get('samples', 'unknown')}
- internal development hold-out: {summary.get('splits', {}).get('internal_development_holdout', {}).get('samples', 'unknown')}
- external periodic test: {summary.get('splits', {}).get('external_periodic_test', {}).get('samples', 'unknown')}

## Packaging Notes

- The exact paper splits are defined by the manifests in `manifests/`.
- The manifests are public-release versions that reference archive names and archive members, not local build paths.
- The Materials Project training-side release uses source corpora plus exact manifests rather than split-specific tarballs.
- Files ending in `inputs_initial_density.tar.gz` contain model inputs.
- Files ending in `targets_dft_density.tar.gz` contain ground-truth DFT charge densities.
- Missing subset archives generated during this packaging run: {generated_names}

## Training Source Corpora

{chr(10).join(f"- `{name}`" for name in train_files)}

Use:

- `manifests/materials_project_train_split.csv`
- `manifests/materials_project_internal_development_holdout_split.csv`

to recover the exact paper samples from those source corpora.

## External Periodic Benchmark Archives

{chr(10).join(f"- `{name}`" for name in external_files)}

## Important Interpretation Note

The internal development hold-out split is included for reproducibility of model
development. It is not the manuscript's external benchmark for out-of-distribution
evaluation.
"""


def build_archive_inventory(archive_entries: list[dict[str, object]]) -> str:
    header = [
        "relative_path",
        "paper_group",
        "subset_name",
        "density_role",
        "sample_count",
        "note",
    ]
    lines = [",".join(header)]
    for entry in archive_entries:
        row = [
            entry["relative_path"],
            entry["paper_group"],
            entry["subset_name"],
            entry["density_role"],
            "" if entry["sample_count"] is None else str(entry["sample_count"]),
            entry["note"].replace(",", ";"),
        ]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def archive_manifest_entry(row: dict[str, str], density_role: str) -> dict[str, str]:
    if row["split_name"] in {"train", "internal_development_holdout"}:
        key = f"training_{density_role}"
        sample_dir = row["parent_id"] if density_role == "input" else row["sample_id"]
    else:
        key = f"{row['subset_name']}_{density_role}"
        sample_dir = row["sample_id"]

    spec = MANIFEST_ARCHIVE_MAP[key]
    member_root = spec["member_root"]
    return {
        "archive": spec["public_archive"],
        "rho_member": f"{member_root}/{sample_dir}/rho_22.npy",
        "grid_member": f"{member_root}/{sample_dir}/grid_sizes_22.dat",
    }


def sanitize_manifest_file(path: Path) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    fieldnames = [
        "split_name",
        "subset_name",
        "sample_id",
        "parent_id",
        "charge",
        "input_archive",
        "input_rho_member",
        "input_grid_member",
        "target_archive",
        "target_rho_member",
        "target_grid_member",
    ]

    sanitized_rows = []
    for row in rows:
        input_entry = archive_manifest_entry(row, "input")
        target_entry = archive_manifest_entry(row, "target")
        sanitized_rows.append(
            {
                "split_name": row["split_name"],
                "subset_name": row["subset_name"],
                "sample_id": row["sample_id"],
                "parent_id": row["parent_id"],
                "charge": row["charge"],
                "input_archive": input_entry["archive"],
                "input_rho_member": input_entry["rho_member"],
                "input_grid_member": input_entry["grid_member"],
                "target_archive": target_entry["archive"],
                "target_rho_member": target_entry["rho_member"],
                "target_grid_member": target_entry["grid_member"],
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanitized_rows)


def infer_present_archive_path(archive_root: Path, archive_name: str) -> Path:
    path = archive_root / archive_name
    if not path.exists():
        raise FileNotFoundError(f"Expected archive not found: {path}")
    return path


def package_release(
    archive_root: Path,
    release_root: Path,
    output_root: Path,
    link_mode: str,
    overwrite: bool,
    generate_missing: bool,
) -> dict[str, object]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    training_dest = output_root / "training_source_corpora"
    external_dest = output_root / "external_periodic_benchmark"
    manifests_dest = output_root / "manifests"
    zenodo_dest = output_root / "zenodo"
    training_dest.mkdir(parents=True, exist_ok=True)
    external_dest.mkdir(parents=True, exist_ok=True)

    archive_entries: list[dict[str, object]] = []
    generated_subsets: list[dict[str, object]] = []

    if generate_missing:
        for spec in MISSING_SUBSET_SPECS:
            generated = build_subset_archives(spec, archive_root, overwrite=overwrite)
            generated_subsets.append(generated)

    for presentation in ARCHIVE_PRESENTATIONS:
        source = infer_present_archive_path(archive_root, presentation.source_name)
        base_dest = training_dest if presentation.group_dir == "training_source_corpora" else external_dest
        dest = base_dest / presentation.output_name
        materialize_existing_file(source, dest, link_mode=link_mode, overwrite=overwrite)
        archive_entries.append(
            {
                "source_name": presentation.source_name,
                "relative_path": str(dest.relative_to(output_root)),
                "output_name": presentation.output_name,
                "paper_group": presentation.paper_group,
                "subset_name": presentation.subset_name,
                "density_role": presentation.density_role,
                "sample_count": presentation.sample_count,
                "note": presentation.note,
            }
        )

    copied_manifest_files = copy_named_files(
        release_root / "manifests",
        manifests_dest,
        MANIFEST_RENAMES,
        overwrite,
    )
    for manifest_path in copied_manifest_files:
        if manifest_path.endswith(".csv"):
            sanitize_manifest_file(Path(manifest_path))
    copied_zenodo_files = copy_named_files(
        release_root / "zenodo",
        zenodo_dest,
        ZENODO_RENAMES,
        overwrite,
    )

    release_summary = load_release_summary(release_root)
    split_summary = release_summary.get("splits", {})
    train_summary = split_summary.get("train", {})
    holdout_summary = split_summary.get("internal_development_holdout", {})
    external_summary = split_summary.get("external_periodic_test", {})

    training_archives = [
        entry["relative_path"]
        for entry in archive_entries
        if entry["paper_group"] == "training_source_corpora"
    ]
    external_archives = [
        {
            "path": entry["relative_path"],
            "subset_name": entry["subset_name"],
            "density_role": entry["density_role"],
            "sample_count": entry["sample_count"],
        }
        for entry in archive_entries
        if entry["paper_group"] == "external_periodic_benchmark"
    ]

    package_summary = {
        "package_name": output_root.name,
        "paper_title": release_summary.get("paper_title"),
        "paper_split_counts": {
            "train": train_summary.get("samples"),
            "internal_development_holdout": holdout_summary.get("samples"),
            "external_periodic_test": external_summary.get("samples"),
        },
        "external_periodic_benchmark_subset_counts": external_summary.get("subsets", {}),
        "training_source_corpora_archives": training_archives,
        "external_periodic_benchmark_archives": external_archives,
        "manifest_files": [str(Path(path).relative_to(output_root)) for path in copied_manifest_files],
        "zenodo_metadata_template": (
            str(Path(copied_zenodo_files[0]).relative_to(output_root))
            if copied_zenodo_files
            else None
        ),
    }
    (output_root / "package_summary.json").write_text(json.dumps(package_summary, indent=2))
    (output_root / "README.md").write_text(
        build_package_readme(release_summary, generated_subsets, archive_entries)
    )
    (output_root / "archive_inventory.csv").write_text(build_archive_inventory(archive_entries))
    return package_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Root containing existing tar.gz archives and where missing subset archives will be created.",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help="Existing chargeflow_jctc_release root containing manifests and Zenodo metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Final publication package output directory.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to materialize existing archives into the final package.",
    )
    parser.add_argument(
        "--skip-generate-missing",
        action="store_true",
        help="Do not generate the missing external subset archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated archives and package files if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = package_release(
        archive_root=args.archive_root,
        release_root=args.release_root,
        output_root=args.output_root,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        generate_missing=not args.skip_generate_missing,
    )
    print(json.dumps(
        {
            "output_root": str(args.output_root),
            "training_archive_count": len(summary["training_source_corpora_archives"]),
            "external_archive_count": len(summary["external_periodic_benchmark_archives"]),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
