#!/usr/bin/env python3
"""
Test script to verify if statistics computation is working.
Run this to diagnose stats engine issues.
"""

import json
import sys

def analyze_stats_output(filepath="output/player_stats.json"):
    """Analyze a stats output file to check what's working."""

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {filepath}")
        return False

    print("=" * 60)
    print("STATS COMPUTATION DIAGNOSTIC")
    print("=" * 60)

    total_players = len(data)
    print(f"\n📊 Total Players: {total_players}")

    if total_players == 0:
        print("❌ No players found in output!")
        return False

    # Analyze first player
    first_player_key = list(data.keys())[0]
    first_player = data[first_player_key]

    print(f"\n🔍 Sample Player: {first_player.get('player_name', 'Unknown')}")
    print(f"   Jersey: {first_player.get('jersey_number', 'Unknown')}")
    print(f"   Team: {first_player.get('team', 'Unknown')}")
    print(f"   Observations: {first_player.get('observations', 0)}")

    # Check what metrics are available
    stats = first_player.get('stats', {})

    print("\n📈 Available Stats Fields:")
    for key in sorted(stats.keys())[:10]:  # Show first 10
        print(f"   • {key}")

    # Check if event stats are working
    print("\n🎯 Event Statistics Check:")

    event_stats = {
        "Ball touches": stats.get('ball_touches_total', 0),
        "Time on ball (s)": stats.get('time_on_ball_s', 0),
        "Distance (m)": stats.get('total_distance', 0),
        "Passes": stats.get('passes_total', 0),
        "Shots on target": stats.get('shots_on_target_total', 0),
        "Shots wide": stats.get('shots_wide_total', 0),
        "Crosses": stats.get('crosses_total', 0),
        "Dribbles": stats.get('dribbles_total', 0),
        "Tackles": stats.get('tackles_total', 0),
        "Interceptions": stats.get('ball_interceptions_total', 0),
        "Goals": stats.get('goals_total', 0),
        "xG (foot)": stats.get('xg_foot_no_opponent', 0),
    }

    working = []
    not_working = []

    for stat_name, value in event_stats.items():
        if value > 0:
            working.append(stat_name)
            print(f"   ✅ {stat_name}: {value}")
        else:
            not_working.append(stat_name)
            print(f"   ❌ {stat_name}: {value}")

    # Aggregate check across all players
    print(f"\n🔬 Checking ALL {total_players} players...")

    all_stats = {
        "passes": 0,
        "shots": 0,
        "crosses": 0,
        "dribbles": 0,
        "tackles": 0,
        "goals": 0,
        "xg": 0.0,
        "ball_touches": 0,
    }

    for player_key, player_data in data.items():
        s = player_data.get('stats', {})
        all_stats["passes"] += s.get('passes_total', 0)
        all_stats["shots"] += s.get('shots_on_target_total', 0) + s.get('shots_wide_total', 0)
        all_stats["crosses"] += s.get('crosses_total', 0)
        all_stats["dribbles"] += s.get('dribbles_total', 0)
        all_stats["tackles"] += s.get('tackles_total', 0)
        all_stats["goals"] += s.get('goals_total', 0)
        all_stats["xg"] += float(s.get('xg_foot_no_opponent', 0) or 0)
        all_stats["ball_touches"] += s.get('ball_touches_total', 0)

    print("\n📊 TOTALS ACROSS ALL PLAYERS:")
    print(f"   Ball touches: {all_stats['ball_touches']}")
    print(f"   Passes: {all_stats['passes']}")
    print(f"   Shots: {all_stats['shots']}")
    print(f"   Crosses: {all_stats['crosses']}")
    print(f"   Dribbles: {all_stats['dribbles']}")
    print(f"   Tackles: {all_stats['tackles']}")
    print(f"   Goals: {all_stats['goals']}")
    print(f"   Total xG: {all_stats['xg']:.2f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT:")
    print("=" * 60)

    if all_stats["ball_touches"] > 0:
        print("✅ Basic tracking is WORKING (ball touches detected)")
    else:
        print("❌ Basic tracking FAILED (no ball touches)")

    if all_stats["passes"] > 0:
        print("✅ Event detection is WORKING (passes detected)")
    else:
        print("❌ Event detection is NOT WORKING (no passes detected)")
        print("\n🔍 Possible causes:")
        print("   1. Pipeline crashed early (check logs)")
        print("   2. Ball tracking failed")
        print("   3. Stats engine not running")
        print("   4. Device/CUDA errors preventing completion")

    if all_stats["xg"] > 0:
        print("✅ xG calculation is WORKING")
    else:
        print("❌ xG calculation is NOT WORKING (no xG detected)")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if all_stats["passes"] == 0:
        print("\n⚠️  CRITICAL: Event statistics are NOT being calculated!")
        print("\nAction steps:")
        print("1. Check pipeline logs: tail -f output/pipeline.log")
        print("2. Look for errors like 'CUDA', 'device', or crashes")
        print("3. Try running on CPU: export CUDA_VISIBLE_DEVICES=''")
        print("4. Test with smaller video: --max_frames 500")
        print("5. Verify ball tracking is working")
        return False
    else:
        print("✅ Stats computation appears to be working correctly!")
        return True

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "output/player_stats.json"
    success = analyze_stats_output(filepath)
    sys.exit(0 if success else 1)
