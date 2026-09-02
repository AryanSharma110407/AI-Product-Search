"""
test_research.py
─────────────────
Quick integration test for the research_agent module.

Run with:
    python test_research.py
"""

import json
import sys
from research_agent import search_product, research_category


def test_single_product():
    """Test single-product structured search."""
    print("=" * 60)
    print("TEST 1: Single Product Search")
    print("=" * 60)

    result = search_product("Logitech MX Master 3S")

    print(f"\n  Query:       {result.query}")
    print(f"  Error:       {result.error}")
    print(f"  Results:     {len(result.results)} found")

    for r in result.results:
        print(f"\n    Platform:    {r.platform}")
        print(f"    Product:     {r.product_name}")
        print(f"    Price:       Rs {r.price_inr:,}")
        print(f"    MRP:         Rs {r.original_mrp:,}" if r.original_mrp else "    MRP:         —")
        print(f"    URL:         {r.source_url or '—'}")
        print(f"    Available:   {r.availability}")

    if result.best_deal:
        bd = result.best_deal
        print(f"\n  Best Deal: {bd.product_name} on {bd.platform} @ Rs {bd.price_inr:,}")

    # Validate structure
    data = result.model_dump()
    assert isinstance(data, dict), "Result should be a dict"
    assert "results" in data, "Result must contain 'results' key"
    assert "query" in data, "Result must contain 'query' key"
    print("\n  Structure validation passed.\n")

    # Print raw JSON for inspection
    print("  Raw JSON output:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return True


def test_category_research():
    """Test multi-candidate category research."""
    print("\n" + "=" * 60)
    print("TEST 2: Category Research (Multi-Candidate)")
    print("=" * 60)

    result = research_category(
        category="24 inch IPS monitors for office",
        count=3,
        budget_hint="under 15000",
    )

    print(f"\n  Category:    {result.category}")
    print(f"  Query Used:  {result.query_used}")
    print(f"  Error:       {result.error}")
    print(f"  Candidates:  {len(result.candidates)} found")

    for c in result.candidates:
        print(f"\n    [{c.platform}] {c.product_name}")
        print(f"      Price:   Rs {c.price_inr:,}")
        if c.specs.ram:
            print(f"      RAM:     {c.specs.ram}")
        if c.specs.screen_size:
            print(f"      Screen:  {c.specs.screen_size}")

    # Validate structure
    data = result.model_dump()
    assert isinstance(data, dict), "Result should be a dict"
    assert "candidates" in data, "Result must contain 'candidates' key"
    assert "category" in data, "Result must contain 'category' key"
    print("\n  Structure validation passed.\n")

    # Print raw JSON for inspection
    print("  Raw JSON output:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return True


if __name__ == "__main__":
    print("\nRunning Research Agent Integration Tests\n")

    passed = 0
    failed = 0

    try:
        if test_single_product():
            passed += 1
    except Exception as e:
        print(f"\n  Test 1 FAILED: {e}")
        failed += 1

    try:
        if test_category_research():
            passed += 1
    except Exception as e:
        print(f"\n  Test 2 FAILED: {e}")
        failed += 1


    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
