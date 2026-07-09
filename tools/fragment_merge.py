"""Fragment-merge + roster mapping + scoring.

Productizes the offline merge prototype: agglomerative clustering of ByteTrack
fragments under hard constraints (no temporal overlap, team-color compatibility,
jersey-conflict separator), then maps clusters -> jersey via pooled reads
constrained by the roster, and scores jersey accuracy/coverage vs the true
roster.

Pipeline: [optional split] -> merge -> map -> score.

Usage:
  python3 -m tools.fragment_merge --input output/<run>/fragments_dump.json \
      --roster roster_babak_gt.json [--split] [--threshold 0.75]

Baseline to beat (Phase 216 remap on Babak clip): 5/12 matched, 10 false.
Only productize into the pipeline if this beats that WITHOUT coverage loss.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

OVERLAP_TOL = 2
STRONG_READS = 3


def _norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def load_fragments(path, min_frames=8):
    """Load fragments. Prefilter tiny/unidentifiable fragments (< min_frames AND
    no jersey reads): they can't be identified or reliably merged, and the O(n^2)
    merge doesn't scale to the raw fragment count on heavily-fragmented clips
    (e.g. 1326 on the Babak training clip vs 289 on HB)."""
    frag = json.load(open(path))
    tracks = {}
    skipped = 0
    for tid, v in frag.items():
        reads = {int(k): c for k, c in (v.get("raw_reads") or {}).items()}
        if v.get("frames", 0) < min_frames and not reads:
            skipped += 1
            continue
        snaps = [_norm(g[1]) for g in (v.get("embedding_gallery") or [])]
        snaps = [s for s in snaps if s is not None]
        if not snaps and v.get("embedding"):
            e = _norm(v["embedding"])
            snaps = [e] if e is not None else []
        strong = {n for n, c in reads.items() if c >= STRONG_READS}
        tracks[tid] = {
            "iv": (v["first_frame"], v["last_frame"]),
            "frames": v.get("frames", 0),
            "color": v.get("color"),
            "reads": reads,
            "gal": snaps,
            "impure": len(strong) > 1,
        }
    if skipped:
        print(f"[prefilter] dropped {skipped} tiny fragments (< {min_frames} frames, no reads) "
              f"-> {len(tracks)} kept")
    return tracks


def merge(tracks, real_teams, threshold):
    ids = list(tracks.keys())

    def overlaps(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) > OVERLAP_TOL

    def team_compatible(ta, tb):
        if ta in real_teams and tb in real_teams:
            return ta == tb
        return True

    def strong_numbers(pooled):
        return {n for n, c in pooled.items() if c >= STRONG_READS}

    clusters = {}
    for i, tid in enumerate(ids):
        t = tracks[tid]
        c = {"members": [tid], "ivs": [t["iv"]], "gal": list(t["gal"]),
             "teams": defaultdict(int), "reads": defaultdict(int),
             "impure": t["impure"]}
        if t["color"] in real_teams:
            c["teams"][t["color"]] += t["frames"]
        for num, cnt in t["reads"].items():
            c["reads"][num] += cnt
        clusters[i] = c

    def compatible(ca, cb):
        if ca["impure"] or cb["impure"]:
            return False
        for a in ca["ivs"]:
            for b in cb["ivs"]:
                if overlaps(a, b):
                    return False
        ta = max(ca["teams"], key=ca["teams"].get) if ca["teams"] else None
        tb = max(cb["teams"], key=cb["teams"].get) if cb["teams"] else None
        if not team_compatible(ta, tb):
            return False
        sa, sb = strong_numbers(ca["reads"]), strong_numbers(cb["reads"])
        if sa and sb and not (sa & sb):
            return False
        return True

    def sim(ca, cb):
        if not ca["gal"] or not cb["gal"]:
            return -1.0
        sims = [float(np.dot(a, b)) for a in ca["gal"] for b in cb["gal"]]
        return 0.65 * (sum(sims) / len(sims)) + 0.35 * max(sims)

    merged = True
    while merged:
        merged = False
        keys = list(clusters.keys())
        best = (threshold, None, None)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = clusters[keys[x]], clusters[keys[y]]
                s = sim(a, b)
                if s <= best[0]:
                    continue
                if compatible(a, b):
                    best = (s, keys[x], keys[y])
        if best[1] is not None:
            a, b = clusters[best[1]], clusters.pop(best[2])
            a["members"] += b["members"]
            a["ivs"] += b["ivs"]
            a["gal"] = (a["gal"] + b["gal"])[:14]
            for t, c in b["teams"].items():
                a["teams"][t] += c
            for num, c in b["reads"].items():
                a["reads"][num] += c
            merged = True
    return list(clusters.values())


def map_and_score(clusters, roster):
    """Assign each cluster a roster jersey via pooled reads (closed-set), enforce
    roster-slot uniqueness, and score matched/false vs the roster."""
    c2t = roster.canonical_color_to_team  # canonical color -> team name

    # score every (cluster, roster-number) pair
    cand = []  # (score, cluster_idx, team, number)
    for ci, c in enumerate(clusters):
        team_color = max(c["teams"], key=c["teams"].get) if c["teams"] else None
        team_name = c2t.get(team_color) if team_color else None
        valid = roster.numbers_for_team(team_name) if team_name else roster.all_numbers()
        if not valid:
            valid = roster.all_numbers()
        for num, cnt in c["reads"].items():
            if num in valid:
                cand.append((cnt, ci, team_name or team_color, num))
    cand.sort(reverse=True)

    assigned_slot = {}   # (team, number) -> cluster_idx
    cluster_num = {}     # cluster_idx -> number
    for score, ci, team, num in cand:
        if ci in cluster_num:
            continue
        slot = (team, num)
        if slot in assigned_slot:
            continue
        assigned_slot[slot] = ci
        cluster_num[ci] = num

    assigned_numbers = set(cluster_num.values())
    return assigned_numbers, cluster_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fragments_dump.json")
    ap.add_argument("--roster", required=True, help="roster JSON (teams + colors)")
    ap.add_argument("--split", action="store_true", help="run split_fragments first")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--min_frames", type=int, default=8,
                    help="drop fragments shorter than this that also have no reads")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from vision.roster import RosterPrior
    roster = RosterPrior.load(args.roster)
    if roster is None:
        print("ERROR: roster failed to load"); sys.exit(1)

    path = args.input
    if args.split:
        from tools.split_fragments import split_fragments
        raw = json.load(open(path))
        split, summary = split_fragments(raw)   # returns (fragments, summary)
        path = args.input.replace(".json", "_split.json")
        json.dump(split, open(path, "w"))
        print(f"[split] {len(raw)} -> {len(split)} fragments "
              f"(split {summary.get('actually_split', 0)}, left impure "
              f"{summary.get('left_impure', 0)}) -> {path}")

    tracks = load_fragments(path, min_frames=args.min_frames)
    real_teams = set(roster.canonical_team_colors().values())
    impure = sum(1 for t in tracks.values() if t["impure"])
    print(f"fragments: {len(tracks)} ({impure} impure) | real teams: {real_teams}")

    clusters = merge(tracks, real_teams, args.threshold)
    assigned, cluster_num = map_and_score(clusters, roster)

    all_gt = roster.all_numbers()
    matched = sorted(assigned & all_gt)
    false = sorted(assigned - all_gt)
    print(f"\nclusters: {len(clusters)} | numbered: {len(cluster_num)}")
    print(f"MATCHED roster numbers: {matched} = {len(matched)}/{len(all_gt)}")
    print(f"FALSE numbers: {false} = {len(false)}")
    print(f"[baseline Phase 216: 5/12 matched, 10 false]")


if __name__ == "__main__":
    main()
