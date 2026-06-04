#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path


def split_words(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return shlex.split(value)


def ext_obj_name(root_dir: Path, source_file: Path) -> str:
    stem_abs = str(source_file.with_suffix("").resolve())
    return str((root_dir / ".csconfig" / "external" / (stem_abs.replace("/", "_") + ".o")).resolve())


def obj_name(root_dir: Path, source_file: Path, build_id: str) -> str:
    rel = source_file.resolve().relative_to(root_dir)
    rel_str = str(rel)
    if rel_str == "src/main.cc":
        return str((root_dir / ".csconfig" / f"{build_id}_main.o").resolve())
    if rel_str.startswith("src/"):
        return str((root_dir / ".csconfig" / rel.with_suffix(".o")).resolve())
    if rel_str.startswith("test/cpp/src/"):
        test_rel = rel.relative_to(Path("test/cpp/src")).with_suffix(".o")
        return str((root_dir / ".csconfig" / "test" / test_rel).resolve())
    if any(rel_str.startswith(prefix) for prefix in ("branch/", "btb/", "prefetcher/", "replacement/")):
        return str((root_dir / ".csconfig" / rel.with_suffix(".o")).resolve())
    return ext_obj_name(root_dir, source_file)


def make_entry(root_dir: Path, source_file: Path, output_file: str, options: list[str]) -> dict:
    args = [os.environ.get("CXX", "g++"), *options, "-I.csconfig", "-c", "-o", output_file, str(source_file.resolve())]
    return {
        "arguments": args,
        "directory": str(root_dir.resolve()),
        "file": str(source_file.resolve()),
        "output": output_file,
    }


def write_db(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)


def generate(root_dir: Path, build_id: str, src_files: list[str], module_files: list[str], test_files: list[str]) -> None:
    src_entries = []
    for file_path in src_files:
        src = Path(file_path)
        src_entries.append(make_entry(root_dir, src, obj_name(root_dir, src, build_id), ["@global.options", "@absolute.options"]))

    test_entries = []
    for file_path in test_files:
        src = Path(file_path)
        test_entries.append(make_entry(root_dir, src, obj_name(root_dir, src, build_id), ["@global.options", "@absolute.options", "-I", ".", "-DCHAMPSIM_TEST_BUILD", "-g3", "-Og"]))

    module_by_dir: dict[Path, list[dict]] = {}
    for file_path in module_files:
        src = Path(file_path)
        module_dir = src.resolve().parent
        entry = make_entry(root_dir, src, obj_name(root_dir, src, build_id), ["@global.options", "@absolute.options", "@module.options"])
        module_by_dir.setdefault(module_dir, []).append(entry)

    write_db(root_dir / "src" / "compile_commands.json", src_entries)
    write_db(root_dir / "test" / "cpp" / "src" / "compile_commands.json", test_entries)
    for module_dir, entries in module_by_dir.items():
        write_db(module_dir / "compile_commands.json", entries)

    # Also write a root database for IDEs expecting this conventional location.
    write_db(root_dir / "compile_commands.json", src_entries + test_entries + [e for entries in module_by_dir.values() for e in entries])


def main():
    parser = argparse.ArgumentParser(description="Generate compile_commands.json")
    parser.add_argument("--build-id", default="TEST")
    parser.add_argument("--root-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--src", default="")
    parser.add_argument("--modules", nargs="*", default=[])
    parser.add_argument("--tests", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    generate(
        root_dir=root_dir,
        build_id=args.build_id if args.build_id else "TEST",
        src_files=split_words(args.src),
        module_files=split_words(args.modules),
        test_files=split_words(args.tests),
    )


if __name__ == "__main__":
    main()
