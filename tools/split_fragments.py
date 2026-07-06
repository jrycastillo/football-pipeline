#!/usr/bin/env python3
import argparse
import copy
import json
import math
from collections import Counter

STRONG_READS = 3
MIN_PURITY = 0.7
MAX_PARTS = 3


def _num_key(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        if value is None:
            return None
        return str(value)


def _event_frame(event):
    try:
        return int(event[0])
    except (TypeError, ValueError, IndexError):
        return None


def _ordered_reads(read_events):
    reads = []
    for event in read_events or []:
        frame = _event_frame(event)
        if frame is None or len(event) < 2:
            continue
        num = _num_key(event[1])
        if num is None:
            continue
        reads.append({"frame": frame, "num": num, "event": list(event)})
    reads.sort(key=lambda r: r["frame"])
    return reads


def _read_counts(reads):
    return Counter(r["num"] for r in reads)


def _strong_numbers(reads, strong_reads=STRONG_READS):
    counts = _read_counts(reads)
    seen = []
    for r in reads:
        if counts[r["num"]] >= strong_reads and r["num"] not in seen:
            seen.append(r["num"])
    return seen


def is_split_candidate(read_events, strong_reads=STRONG_READS):
    reads = _ordered_reads(read_events)
    return len(_strong_numbers(reads, strong_reads)) >= 2


def _best_pair_split(reads, strong_reads=STRONG_READS, min_purity=MIN_PURITY):
    strong_nums = _strong_numbers(reads, strong_reads)
    if len(strong_nums) < 2:
        return None

    best = None
    for left_num in strong_nums:
        for right_num in strong_nums:
            if left_num == right_num:
                continue
            filtered = [r for r in reads if r["num"] in (left_num, right_num)]
            if len(filtered) < 4:
                continue
            for cut in range(2, len(filtered) - 1):
                left = filtered[:cut]
                right = filtered[cut:]
                left_purity = sum(1 for r in left if r["num"] == left_num) / len(left)
                right_purity = sum(1 for r in right if r["num"] == right_num) / len(right)
                if left_purity < min_purity or right_purity < min_purity:
                    continue

                left_target = [r["frame"] for r in left if r["num"] == left_num]
                right_target = [r["frame"] for r in right if r["num"] == right_num]
                if not left_target or not right_target:
                    continue
                last_left = max(left_target)
                first_right = min(right_target)
                if last_left >= first_right:
                    continue

                score = left_purity + right_purity
                split = {
                    "score": score,
                    "min_purity": min(left_purity, right_purity),
                    "boundary": (last_left + first_right) / 2.0,
                    "left_num": left_num,
                    "right_num": right_num,
                    "left_purity": left_purity,
                    "right_purity": right_purity,
                }
                if best is None:
                    best = split
                elif (split["score"], split["min_purity"]) > (best["score"], best["min_purity"]):
                    best = split
    return best


def _fragment_extent(fragment, reads, gallery):
    frames = [r["frame"] for r in reads]
    frames.extend(_event_frame(g) for g in gallery or [])
    frames = [f for f in frames if f is not None]

    first = fragment.get("first_frame")
    last = fragment.get("last_frame")
    try:
        first = int(first)
    except (TypeError, ValueError):
        first = min(frames) if frames else 0
    try:
        last = int(last)
    except (TypeError, ValueError):
        last = max(frames) if frames else first
    if last < first:
        first, last = last, first
    return first, last


def _split_segments(reads, first_frame, last_frame, strong_reads, min_purity, max_parts):
    candidate = len(_strong_numbers(reads, strong_reads)) >= 2
    if not candidate:
        return [{"first_frame": first_frame, "last_frame": last_frame,
                 "reads": reads, "impure": False}]
    if max_parts <= 1:
        return [{"first_frame": first_frame, "last_frame": last_frame,
                 "reads": reads, "impure": True}]

    split = _best_pair_split(reads, strong_reads=strong_reads, min_purity=min_purity)
    if split is None:
        return [{"first_frame": first_frame, "last_frame": last_frame,
                 "reads": reads, "impure": True}]

    left_last = int(math.floor(split["boundary"]))
    left_last = max(first_frame, min(left_last, last_frame - 1))
    right_first = left_last + 1
    if right_first > last_frame:
        return [{"first_frame": first_frame, "last_frame": last_frame,
                 "reads": reads, "impure": True}]

    left_reads = [r for r in reads if r["frame"] <= left_last]
    right_reads = [r for r in reads if r["frame"] >= right_first]
    if len(left_reads) < 2 or len(right_reads) < 2:
        return [{"first_frame": first_frame, "last_frame": last_frame,
                 "reads": reads, "impure": True}]

    left_parts = _split_segments(
        left_reads, first_frame, left_last, strong_reads, min_purity, max_parts - 1)
    remaining = max(1, max_parts - len(left_parts))
    right_parts = _split_segments(
        right_reads, right_first, last_frame, strong_reads, min_purity, remaining)
    return left_parts + right_parts


def _raw_reads(reads):
    counts = Counter()
    for r in reads:
        counts[r["num"]] += 1
    return {k: int(v) for k, v in counts.items()}


def _gallery_for_range(gallery, first_frame, last_frame):
    out = []
    for entry in gallery or []:
        frame = _event_frame(entry)
        if frame is not None and first_frame <= frame <= last_frame:
            out.append(copy.deepcopy(entry))
    return out


def _mean_normalized_embedding(gallery):
    vectors = []
    for entry in gallery or []:
        if len(entry) < 2 or not isinstance(entry[1], list) or not entry[1]:
            continue
        try:
            vectors.append([float(x) for x in entry[1]])
        except (TypeError, ValueError):
            continue
    if not vectors:
        return None

    dim = min(len(v) for v in vectors)
    if dim <= 0:
        return None
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm <= 0:
        return None
    return [round(x / norm, 5) for x in mean]


def _prorated_frames(original_frames, original_first, original_last, part_first, part_last):
    try:
        original_frames = int(original_frames)
    except (TypeError, ValueError):
        original_frames = max(1, original_last - original_first + 1)
    original_extent = max(1, original_last - original_first + 1)
    part_extent = max(1, part_last - part_first + 1)
    return max(1, int(round(original_frames * (part_extent / original_extent))))


def _build_split_part(fragment, segment, gallery, original_first, original_last):
    part = copy.deepcopy(fragment)
    part.pop("votes", None)
    part.pop("impure", None)
    part["first_frame"] = int(segment["first_frame"])
    part["last_frame"] = int(segment["last_frame"])
    part["frames"] = _prorated_frames(
        fragment.get("frames"),
        original_first,
        original_last,
        part["first_frame"],
        part["last_frame"],
    )
    part["read_events"] = [copy.deepcopy(r["event"]) for r in segment["reads"]]
    part["raw_reads"] = _raw_reads(segment["reads"])
    part["embedding_gallery"] = _gallery_for_range(
        gallery, part["first_frame"], part["last_frame"])
    part["embedding"] = _mean_normalized_embedding(part["embedding_gallery"])
    if segment.get("impure"):
        part["impure"] = True
    return part


def split_fragment(tid, fragment, strong_reads=STRONG_READS,
                   min_purity=MIN_PURITY, max_parts=MAX_PARTS):
    reads = _ordered_reads(fragment.get("read_events", []))
    if len(_strong_numbers(reads, strong_reads)) < 2:
        return [(str(tid), copy.deepcopy(fragment))], False, False

    gallery = fragment.get("embedding_gallery", [])
    first_frame, last_frame = _fragment_extent(fragment, reads, gallery)
    segments = _split_segments(
        reads, first_frame, last_frame, strong_reads, min_purity, max_parts)

    if len(segments) == 1:
        out = copy.deepcopy(fragment)
        out["impure"] = True
        return [(str(tid), out)], True, False

    parts = []
    for idx, segment in enumerate(segments):
        key = f"{tid}{chr(ord('a') + idx)}"
        parts.append((key, _build_split_part(
            fragment, segment, gallery, first_frame, last_frame)))
    return parts, True, True


def split_fragments(fragments, strong_reads=STRONG_READS, min_purity=MIN_PURITY):
    if not isinstance(fragments, dict):
        raise ValueError("fragments input must be a JSON object keyed by track id")

    out = {}
    summary = {
        "fragments_in": len(fragments),
        "split_candidates": 0,
        "actually_split": 0,
        "left_impure": 0,
    }
    for tid, fragment in fragments.items():
        if not isinstance(fragment, dict):
            out[str(tid)] = copy.deepcopy(fragment)
            continue
        parts, candidate, did_split = split_fragment(
            tid, fragment, strong_reads=strong_reads, min_purity=min_purity)
        if candidate:
            summary["split_candidates"] += 1
        if did_split:
            summary["actually_split"] += 1
        elif candidate:
            summary["left_impure"] += 1
        for key, part in parts:
            out[key] = part
    return out, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split fragment dumps at likely mid-track identity switches.")
    parser.add_argument("--input", required=True, help="Input fragments_dump.json")
    parser.add_argument("--output", required=True, help="Output fragments_split.json")
    parser.add_argument("--strong_reads", type=int, default=STRONG_READS)
    parser.add_argument("--min_purity", type=float, default=MIN_PURITY)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r") as f:
        fragments = json.load(f)

    split, summary = split_fragments(
        fragments,
        strong_reads=args.strong_reads,
        min_purity=args.min_purity,
    )
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print(
        "fragments in: {fragments_in}, split candidates: {split_candidates}, "
        "actually split: {actually_split}, left impure: {left_impure}".format(**summary)
    )


if __name__ == "__main__":
    main()
