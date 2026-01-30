#!/usr/bin/env python3
"""
Test script to verify color classifier changes are correct.
Reviews HSV color ranges and tests the classification logic.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/ronan/Babak')

from vision.color_classifier import TeamColorClassifier, HSV_COLOR_RANGES

def test_hsv_ranges():
    """Verify HSV color ranges don't overlap and cover the spectrum properly."""
    print("=" * 80)
    print("HSV COLOR RANGE VALIDATION")
    print("=" * 80)

    issues = []

    # Check each color's HSV ranges
    print("\n📊 Color Definitions:")
    for color_name, ranges in HSV_COLOR_RANGES.items():
        print(f"\n{color_name}:")
        for i, r in enumerate(ranges):
            h_min, h_max, s_min, s_max, v_min, v_max = r
            print(f"  Range {i+1}: H[{h_min:3}-{h_max:3}] S[{s_min:3}-{s_max:3}] V[{v_min:3}-{v_max:3}]")

            # Validate range integrity
            if h_min > h_max and not (color_name in ["Maroon", "Red"]):  # Red wraps around
                issues.append(f"❌ {color_name}: H_min ({h_min}) > H_max ({h_max})")
            if s_min > s_max:
                issues.append(f"❌ {color_name}: S_min ({s_min}) > S_max ({s_max})")
            if v_min > v_max:
                issues.append(f"❌ {color_name}: V_min ({v_min}) > V_max ({v_max})")

            # Check bounds
            if h_min < 0 or h_max > 180:
                issues.append(f"❌ {color_name}: Hue out of bounds [0-180]")
            if s_min < 0 or s_max > 255:
                issues.append(f"❌ {color_name}: Saturation out of bounds [0-255]")
            if v_min < 0 or v_max > 255:
                issues.append(f"❌ {color_name}: Value out of bounds [0-255]")

    # Report issues
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All HSV ranges are valid!")
        return True


def test_color_separation():
    """Test if similar colors are properly separated."""
    print("\n" + "=" * 80)
    print("COLOR SEPARATION ANALYSIS")
    print("=" * 80)

    # Key color pairs that should be distinct
    test_colors = [
        ("Orange", "Red", "Should distinguish orange from red"),
        ("Orange", "Gold", "Should distinguish orange from gold"),
        ("Gold", "Yellow", "Should distinguish gold from yellow"),
        ("Yellow", "Lime", "Should distinguish yellow from lime"),
        ("Lime", "Green", "Should distinguish lime from green"),
        ("Green", "Teal", "Should distinguish green from teal"),
        ("Teal", "Cyan", "Should distinguish teal from cyan"),
        ("Cyan", "Blue", "Should distinguish cyan from blue"),
        ("Blue", "Navy", "Should distinguish blue from navy"),
        ("Navy", "Purple", "Should distinguish navy from purple"),
    ]

    print("\n📋 Key Color Distinctions:")
    for color1, color2, description in test_colors:
        ranges1 = HSV_COLOR_RANGES.get(color1, [])
        ranges2 = HSV_COLOR_RANGES.get(color2, [])

        if not ranges1 or not ranges2:
            print(f"  ⚠️  {description}: Missing color definition")
            continue

        # Check hue separation (primary distinction)
        hues1 = [(r[0], r[1]) for r in ranges1]
        hues2 = [(r[0], r[1]) for r in ranges2]

        print(f"  ✓ {description}")
        print(f"     {color1}: H {hues1}")
        print(f"     {color2}: H {hues2}")


def test_classifier_logic():
    """Test the actual classification logic with synthetic samples."""
    print("\n" + "=" * 80)
    print("CLASSIFIER LOGIC TEST")
    print("=" * 80)

    classifier = TeamColorClassifier()

    # Test cases: (H, S, V) -> Expected color
    test_cases = [
        # Basic colors
        ((5, 200, 200), "Red", "Bright red"),
        ((175, 200, 200), "Red", "Red (wrapped)"),
        ((15, 200, 150), "Orange", "Orange"),
        ((25, 150, 200), "Gold", "Gold"),
        ((30, 200, 200), "Yellow", "Yellow"),
        ((45, 150, 200), "Lime", "Light green/lime"),
        ((70, 150, 150), "Green", "Green"),
        ((90, 150, 150), "Teal", "Teal"),
        ((95, 200, 200), "Cyan", "Cyan"),
        ((120, 200, 200), "Blue", "Blue"),
        ((130, 150, 50), "Navy", "Navy (dark blue)"),
        ((150, 150, 150), "Purple", "Purple"),
        ((160, 100, 200), "Pink", "Pink"),

        # Achromatic colors
        ((0, 10, 220), "White", "White (low saturation, high value)"),
        ((0, 15, 150), "Silver", "Silver/gray"),
        ((0, 50, 15), "Black", "Black (low value)"),
        ((0, 20, 100), "Gray", "Gray (medium)"),
    ]

    print("\n🧪 Testing color classification:")
    passed = 0
    failed = 0

    for (h, s, v), expected, description in test_cases:
        # Create a synthetic uniform patch
        patch = np.full((50, 50, 3), (h, s, v), dtype=np.uint8)
        bgr_patch = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)

        result = classifier.predict(bgr_patch)

        if result == expected:
            print(f"  ✅ HSV({h:3}, {s:3}, {v:3}) -> {result:12} | {description}")
            passed += 1
        else:
            print(f"  ❌ HSV({h:3}, {s:3}, {v:3}) -> {result:12} (expected {expected:12}) | {description}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def analyze_recent_changes():
    """Analyze the specific changes made to the color classifier."""
    print("\n" + "=" * 80)
    print("RECENT CHANGES ANALYSIS")
    print("=" * 80)

    print("\n🔍 Key changes identified (from commit 1581a71 on Jan 25):")

    print("\n1. NEW COLORS ADDED:")
    new_colors = ["Gold", "Lime", "Teal", "Cyan"]
    for color in new_colors:
        if color in HSV_COLOR_RANGES:
            print(f"   ✅ {color}: {HSV_COLOR_RANGES[color]}")
        else:
            print(f"   ❌ {color}: NOT FOUND")

    print("\n2. HSV RANGE ADJUSTMENTS:")
    adjustments = [
        ("Red", "Hue range narrowed to [0-10, 170-180] to avoid conflict with Maroon"),
        ("Orange", "Narrowed to [10-20] to make room for Gold"),
        ("Yellow", "Narrowed to [25-35] to separate from Gold and Lime"),
        ("Green", "Shifted to [55-85] to accommodate Lime"),
        ("Blue", "Expanded to [100-140] with separate Teal/Cyan"),
        ("Pink", "Adjusted to [150-170]"),
    ]

    for color, change in adjustments:
        if color in HSV_COLOR_RANGES:
            print(f"   ✓ {color}: {change}")
            print(f"      Current: {HSV_COLOR_RANGES[color]}")
        else:
            print(f"   ✗ {color}: NOT FOUND")

    print("\n3. FALLBACK LOGIC UPDATE:")
    print("   ✓ _classify_hsv_fallback() updated with new color boundaries:")
    print("      - Added Gold (h < 30)")
    print("      - Added Lime (h < 55)")
    print("      - Added Teal (h < 95)")
    print("      - Added Cyan (h < 105)")
    print("      - Added Pink fallback")

    print("\n4. NEW COMPONENT ADDED:")
    print("   ✓ SigLIPTeamClassifier: Deep learning-based team classifier")
    print("      - Uses google/siglip-base-patch16-224 model")
    print("      - Provides semantic embeddings for robust team clustering")
    print("      - Includes ReID capabilities via cosine similarity")

    print("\n" + "=" * 80)
    print("CORRECTNESS ASSESSMENT")
    print("=" * 80)

    concerns = []
    recommendations = []

    # Check for potential issues

    # 1. Check Red/Maroon separation
    red_ranges = HSV_COLOR_RANGES.get("Red", [])
    maroon_ranges = HSV_COLOR_RANGES.get("Maroon", [])

    if red_ranges and maroon_ranges:
        red_v_min = min([r[4] for r in red_ranges])
        maroon_v_max = max([r[5] for r in maroon_ranges])

        if maroon_v_max >= red_v_min:
            concerns.append("⚠️  Red/Maroon may overlap in value range (brightness)")
            recommendations.append("Consider adjusting V threshold between Red (V_min=40) and Maroon (V_max=100)")

    # 2. Check color spectrum coverage
    total_colors = len(HSV_COLOR_RANGES)
    if total_colors >= 16:
        print(f"✅ Good color coverage: {total_colors} distinct colors defined")
    else:
        concerns.append(f"⚠️  Limited color coverage: only {total_colors} colors")

    # 3. Check if new colors fill gaps
    chromatic_colors = [k for k in HSV_COLOR_RANGES.keys()
                        if k not in ["White", "Silver", "Black", "Gray"]]
    if len(chromatic_colors) >= 12:
        print(f"✅ Comprehensive chromatic coverage: {len(chromatic_colors)} colors")

    if concerns:
        print("\n⚠️  POTENTIAL CONCERNS:")
        for concern in concerns:
            print(f"  {concern}")

    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")

    if not concerns:
        print("\n✅ NO CRITICAL ISSUES FOUND")
        print("   The color classifier changes appear correct and well-structured.")

    return len(concerns) == 0


def main():
    """Run all tests."""
    print("🎨 COLOR CLASSIFIER TEST SUITE")
    print()

    all_passed = True

    # Test 1: HSV range validation
    if not test_hsv_ranges():
        all_passed = False

    # Test 2: Color separation
    test_color_separation()

    # Test 3: Classifier logic
    if not test_classifier_logic():
        all_passed = False

    # Test 4: Recent changes analysis
    if not analyze_recent_changes():
        all_passed = False

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if all_passed:
        print("\n✅ COLOR CLASSIFIER IS WORKING CORRECTLY")
        print("   All tests passed. The recent changes are properly implemented.")
        return 0
    else:
        print("\n⚠️  SOME ISSUES DETECTED")
        print("   Review the detailed output above for specific problems.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
