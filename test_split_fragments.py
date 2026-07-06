#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile

from tools.split_fragments import split_fragments


def _fragment(first=0, last=100, frames=101, reads=None, gallery=None):
    return {
        "first_frame": first,
        "last_frame": last,
        "frames": frames,
        "color": "Red",
        "embedding": [0.5, 0.5],
        "embedding_gallery": gallery or [],
        "raw_reads": {},
        "read_events": reads or [],
        "votes": {"10": 1.0},
    }


def test_clean_switch_splits_at_midpoint():
    frag = _fragment(
        reads=[
            [10, 10, 0.9], [20, 10, 0.9], [30, 10, 0.9],
            [60, 20, 0.9], [70, 20, 0.9], [80, 20, 0.9],
        ],
        gallery=[
            [15, [1.0, 0.0]],
            [65, [0.0, 1.0]],
        ],
    )

    out, summary = split_fragments({"12": frag})

    assert list(out.keys()) == ["12a", "12b"]
    assert summary == {
        "fragments_in": 1,
        "split_candidates": 1,
        "actually_split": 1,
        "left_impure": 0,
    }
    assert out["12a"]["first_frame"] == 0
    assert out["12a"]["last_frame"] == 45
    assert out["12b"]["first_frame"] == 46
    assert out["12b"]["last_frame"] == 100
    assert out["12a"]["raw_reads"] == {"10": 3}
    assert out["12b"]["raw_reads"] == {"20": 3}
    assert out["12a"]["embedding_gallery"] == [[15, [1.0, 0.0]]]
    assert out["12b"]["embedding_gallery"] == [[65, [0.0, 1.0]]]
    assert out["12a"]["embedding"] == [1.0, 0.0]
    assert out["12b"]["embedding"] == [0.0, 1.0]
    assert "votes" not in out["12a"]
    assert "votes" not in out["12b"]
    assert out["12a"]["color"] == "Red"


def test_interleaved_reads_mark_impure_without_split():
    frag = _fragment(
        reads=[
            [10, 10, 0.9], [20, 20, 0.9], [30, 10, 0.9],
            [40, 20, 0.9], [50, 10, 0.9], [60, 20, 0.9],
        ]
    )

    out, summary = split_fragments({"7": frag})

    assert list(out.keys()) == ["7"]
    assert out["7"]["impure"] is True
    assert out["7"]["read_events"] == frag["read_events"]
    assert out["7"]["votes"] == frag["votes"]
    assert summary["split_candidates"] == 1
    assert summary["actually_split"] == 0
    assert summary["left_impure"] == 1


def test_single_number_fragment_is_untouched():
    frag = _fragment(
        reads=[[10, 10, 0.9], [20, 10, 0.9], [30, 10, 0.9]],
        gallery=[[15, [1.0, 0.0]]],
    )

    out, summary = split_fragments({"3": frag})

    assert out == {"3": frag}
    assert summary == {
        "fragments_in": 1,
        "split_candidates": 0,
        "actually_split": 0,
        "left_impure": 0,
    }


def test_three_way_switch_splits_recursively():
    frag = _fragment(
        first=0,
        last=120,
        frames=121,
        reads=[
            [10, 10, 0.9], [20, 10, 0.9], [30, 10, 0.9],
            [50, 20, 0.9], [60, 20, 0.9], [70, 20, 0.9],
            [90, 30, 0.9], [100, 30, 0.9], [110, 30, 0.9],
        ],
        gallery=[
            [15, [1.0, 0.0, 0.0]],
            [55, [0.0, 1.0, 0.0]],
            [95, [0.0, 0.0, 1.0]],
        ],
    )

    out, summary = split_fragments({"5": frag})

    assert list(out.keys()) == ["5a", "5b", "5c"]
    assert summary["split_candidates"] == 1
    assert summary["actually_split"] == 1
    assert out["5a"]["first_frame"] == 0
    assert out["5a"]["last_frame"] == 40
    assert out["5b"]["first_frame"] == 41
    assert out["5b"]["last_frame"] == 80
    assert out["5c"]["first_frame"] == 81
    assert out["5c"]["last_frame"] == 120
    assert out["5a"]["raw_reads"] == {"10": 3}
    assert out["5b"]["raw_reads"] == {"20": 3}
    assert out["5c"]["raw_reads"] == {"30": 3}
    assert out["5a"]["embedding_gallery"] == [[15, [1.0, 0.0, 0.0]]]
    assert out["5b"]["embedding_gallery"] == [[55, [0.0, 1.0, 0.0]]]
    assert out["5c"]["embedding_gallery"] == [[95, [0.0, 0.0, 1.0]]]
    assert out["5a"]["frames"] == 41
    assert out["5b"]["frames"] == 40
    assert out["5c"]["frames"] == 40


def test_cli_writes_split_json_and_summary():
    frag = _fragment(
        reads=[
            [10, 10, 0.9], [20, 10, 0.9], [30, 10, 0.9],
            [60, 20, 0.9], [70, 20, 0.9], [80, 20, 0.9],
        ]
    )

    with tempfile.TemporaryDirectory() as td:
        input_path = os.path.join(td, "fragments_dump.json")
        output_path = os.path.join(td, "fragments_split.json")
        with open(input_path, "w") as f:
            json.dump({"12": frag}, f)

        result = subprocess.run(
            [
                sys.executable,
                "tools/split_fragments.py",
                "--input", input_path,
                "--output", output_path,
            ],
            cwd=os.path.dirname(__file__) or ".",
            check=True,
            text=True,
            capture_output=True,
        )

        with open(output_path) as f:
            written = json.load(f)

    assert list(written.keys()) == ["12a", "12b"]
    assert "fragments in: 1" in result.stdout
    assert "split candidates: 1" in result.stdout
    assert "actually split: 1" in result.stdout
    assert "left impure: 0" in result.stdout


if __name__ == "__main__":
    test_clean_switch_splits_at_midpoint()
    test_interleaved_reads_mark_impure_without_split()
    test_single_number_fragment_is_untouched()
    test_three_way_switch_splits_recursively()
    test_cli_writes_split_json_and_summary()
    print("test_split_fragments passed")
