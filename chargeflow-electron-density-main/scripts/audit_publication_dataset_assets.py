#!/usr/bin/env python3
"""Audit dataset assets for the ChargeFlow publication release.

This script cross-checks three things:
1. the train/test list files used by the current code,
2. the existing publication manifests in `chargeflow_jctc_release`,
3. the prebuilt tar archives in `/vast/.../Test-set-charge-electron-density`.

It writes a JSON summary that can be used as a release checklist.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path


FINAL_CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FINAL_CODE_ROOT.parent
TAR_ROOT = Path("/vast/minhtrin/Charged_electron_density/Test-set-charge-electron-density")
RELEASE_ROOT = Path("/vast/minhtrin/Charged_electron_density/dataset/chargeflow_jctc_release")
DEFAULT_OUTPUT = FINAL_CODE_ROOT / "data_lists" / "stats" / "publication_asset_audit.json"


TRAIN_LISTS = {
    "train_data": FINAL_CODE_ROOT / "data_lists" / "training" / "list_d_subMP_12k_charged_train",
    "train_label": FINAL_CODE_ROOT / "data_lists" / "training" / "list_l_subMP_12k_charged_train",
    "holdout_data": FINAL_CODE_ROOT / "data_lists" / "training" / "list_d_subMP_12k_charged_test",
    "holdout_label": FINAL_CODE_ROOT / "data_lists" / "training" / "list_l_subMP_12k_charged_test",
}


TEST_ARCHIVES = {
    "perovskites": (
        "perovskites_rho.tar.gz",
        "perovskites_rho_initial.tar.gz",
    ),
    "charged_defects": (
        "defects_charged_rho.tar.gz",
        "defects_charged_rho_initial.tar.gz",
    ),
    "multisite_defects_diamond": (
        "MultisiteDefects_Diamond_rho.tar.gz",
        "MultisiteDefects_Diamond_rho_initial.tar.gz",
    ),
    "special_defects_diamond": (
        "SpecialDefects_Diamond_rho.tar.gz",
        "SpecialDefects_Diamond_rho_initial.tar.gz",
    ),
    "mofs": (
        "mofs_rho.tar.gz",
        "mofs_rho_initial.tar.gz",
    ),
    "extreme_mofs": (
        "extreme_mofs_rho.tar.gz",
        "extreme_mofs_rho_initial.tar.gz",
    ),
    "organic": (
        "organic_rho.tar.gz",
        "organic_rho_initial.tar.gz",
    ),
    "extreme_organic": (
        "extreme_organic_rho.tar.gz",
        "extreme_organic_rho_initial.tar.gz",
    ),
}


TRAINING_ARCHIVES = {
    "paper_train_inputs": "MP_neutral_rho_initial.tar.gz",
    "paper_train_targets": "subMP12k_charged_rho.tar.gz",
    "extra_mp_charge_initial": "MP_charge_rho_initial.tar.gz",
    "extra_mp_neutral_targets": "MP_neutral_rho.tar.gz",
}


def read_nonempty_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def sample_count_in_tar(path: Path) -> int:
    count = 0
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            parts = member.name.split("/")
            if len(parts) == 2 and member.isdir():
                count += 1
    return count


def first_archive_entries(path: Path, limit: int = 6) -> list[str]:
    entries: list[str] = []
    with tarfile.open(path, "r:gz") as tf:
        for _ in range(limit):
            member = tf.next()
            if member is None:
                break
            entries.append(member.name)
    return entries


def train_root_from_list(path: Path) -> str:
    lines = read_nonempty_lines(path)
    if not lines:
        return ""
    first = Path(lines[0])
    if len(first.parts) < 2:
        return str(first)
    return first.parents[1].name


def expected_external_counts() -> dict[str, int]:
    root = FINAL_CODE_ROOT / "data_lists" / "test_sets"
    counts = {}
    for subset in [
        "perovskites",
        "multisite_defects_diamond",
        "special_defects_diamond",
        "mofs",
        "extreme_mofs",
        "organic",
        "extreme_organic",
    ]:
        counts[subset] = len(read_nonempty_lines(root / f"list_l_{subset}"))
    counts["charged_defects"] = len(
        read_nonempty_lines(FINAL_CODE_ROOT / "data_lists" / "training" / "list_l_defects_charged")
    )
    return counts


def load_release_summary() -> dict:
    summary_path = RELEASE_ROOT / "manifests" / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def build_audit() -> dict:
    train_list_summary = {}
    for key, path in TRAIN_LISTS.items():
        lines = read_nonempty_lines(path)
        train_list_summary[key] = {
            "path": str(path),
            "count": len(lines),
            "source_root_name": train_root_from_list(path),
            "first_entries": lines[:3],
        }

    expected_counts = expected_external_counts()

    tar_summary = {}
    for label, archive_name in TRAINING_ARCHIVES.items():
        archive_path = TAR_ROOT / archive_name
        tar_summary[label] = {
            "archive_path": str(archive_path),
            "exists": archive_path.exists(),
            "sample_count": sample_count_in_tar(archive_path) if archive_path.exists() else None,
            "first_entries": first_archive_entries(archive_path) if archive_path.exists() else [],
        }

    external_archive_summary = {}
    present_pairs = []
    missing_pairs = []
    for subset, (target_archive, input_archive) in TEST_ARCHIVES.items():
        target_path = TAR_ROOT / target_archive
        input_path = TAR_ROOT / input_archive
        both_exist = target_path.exists() and input_path.exists()
        if both_exist:
            present_pairs.append(subset)
        else:
            missing_pairs.append(subset)
        external_archive_summary[subset] = {
            "target_archive": str(target_path),
            "input_archive": str(input_path),
            "target_exists": target_path.exists(),
            "input_exists": input_path.exists(),
            "expected_count": expected_counts[subset],
            "target_sample_count": sample_count_in_tar(target_path) if target_path.exists() else None,
            "input_sample_count": sample_count_in_tar(input_path) if input_path.exists() else None,
        }

    release_summary = load_release_summary()

    return {
        "train_list_summary": train_list_summary,
        "release_summary": release_summary,
        "training_archive_summary": tar_summary,
        "external_archive_summary": external_archive_summary,
        "conclusions": {
            "paper_train_inputs_archive": "MP_neutral_rho_initial.tar.gz",
            "paper_train_targets_archive": "subMP12k_charged_rho.tar.gz",
            "paper_train_archives_are_split_specific": False,
            "external_test_subsets_reusable_directly": present_pairs,
            "external_test_subsets_missing_from_tar_folder": missing_pairs,
            "recommended_publication_strategy": [
                "Reuse the existing MP tar archives as source corpora for the train and internal development hold-out splits, but publish exact split manifests from the paper rather than the unsplit MP tarballs alone.",
                "Reuse the existing tarballs for perovskites, charged defects, multisite defects diamond, and special defects diamond.",
                "Generate the missing MOF, extreme MOF, organic, and extreme organic tarballs to complete the external periodic benchmark.",
                "Ship the final publication record with README/manifests from chargeflow_jctc_release so the archived files are explicitly tied to the manuscript splits.",
            ],
        },
    }


def main() -> int:
    audit = build_audit()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(audit, indent=2))
    print(f"Wrote audit to {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
