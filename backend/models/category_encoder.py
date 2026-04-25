"""
models/category_encoder.py — Map raw StockCode descriptions → product categories.

Uses keyword matching on the Description field to assign ~15 business-meaningful
categories. This reduces the state space from 4,623 raw SKUs to ~15 categories, 
making the Markov transition matrix dense enough to yield useful predictions.

Category hierarchy (ordered by priority — first match wins):
  CANDLES_LIGHTS   → wax, candle, lantern, tealight
  FRAMES_MIRRORS   → frame, mirror, photo
  BAGS_STORAGE     → bag, purse, trunk, chest, box, tin, basket
  KITCHEN_DINING   → mug, cup, bowl, plate, cake, kitchen, jug, teapot
  GARDEN_OUTDOOR   → garden, planter, birdhouse, watering
  CHRISTMAS        → christmas, xmas, santa, reindeer, snowman, holly
  SIGNS_PLAQUES    → sign, plaque, metal wall
  CHILDREN_TOYS    → children, kids, toy, game, doll
  STATIONERY       → notebook, pencil, pen, ruler, stationery
  TEXTILES_SOFT    → cushion, blanket, throw, pillow
  CLOCKS_TIMEPIECE → clock, alarm, timer
  JEWELLERY_ACCESS → necklace, bracelet, earring, ring, jewel
  SEASONAL_GIFT    → gift, wrap, ribbon, tag, card, birthday
  HOME_DECOR       → heart, bunting, garland, wreath, decorat
  OTHER            → catch-all
"""
from __future__ import annotations

import re
import pandas as pd

# ── Category rules (ordered — first match wins) ───────────────────────────────

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("CHRISTMAS",        ["christmas", "xmas", "santa", "reindeer", "snowman", "holly", "advent"]),
    ("CANDLES_LIGHTS",   ["candle", "wax", "lantern", "tealight", "t-light", "tea light",
                          "fairy light", "nightlight", "light holder", "t light"]),
    ("FRAMES_MIRRORS",   ["frame", "mirror", "photo", "picture"]),
    ("KITCHEN_DINING",   ["mug", "cup", "bowl", "plate", "cake", "kitchen", "jug", "teapot",
                          "coffee", "tea set", "cutlery", "cookbook", "apron", "baking"]),
    ("BAGS_STORAGE",     ["bag", "purse", "trunk", "chest", "box", "tin", "basket",
                          "suitcase", "wallet", "pouch", "hamper"]),
    ("GARDEN_OUTDOOR",   ["garden", "planter", "birdhouse", "watering", "outdoor",
                          "bird", "flower pot", "seed", "fence"]),
    ("CHILDREN_TOYS",    ["children", "kids", "toy", "game", "doll", "puppet", "animal",
                          "baby", "teddy", "puzzle"]),
    ("SIGNS_PLAQUES",    ["sign", "plaque", "metal wall", "vintage label", "door sign"]),
    ("TEXTILES_SOFT",    ["cushion", "blanket", "throw", "pillow", "quilt", "linen",
                          "napkin", "tablecloth"]),
    ("CLOCKS_TIMEPIECE", ["clock", "alarm", "timer", "watch"]),
    ("JEWELLERY_ACCESS", ["necklace", "bracelet", "earring", "ring", "jewel", "bangle",
                          "brooch", "charm", "pendant"]),
    ("STATIONERY",       ["notebook", "pencil", "pen ", "ruler", "stationery",
                          "diary", "journal", "notepad"]),
    ("SEASONAL_GIFT",    ["gift", "wrap", "ribbon", "tag", "card", "birthday",
                          "present", "party", "balloon", "celebration"]),
    ("HOME_DECOR",       ["heart", "bunting", "garland", "wreath", "decorat", "ornament",
                          "vase", "tray", "shelf", "wall art", "vintage", "retro",
                          "painted", "floral", "rose", "antique"]),
]

# Pre-compile patterns for speed
_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (cat, re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE))
    for cat, keywords in CATEGORY_RULES
]

FALLBACK_CATEGORY = "OTHER"


# ── Core function ──────────────────────────────────────────────────────────────

def assign_category(description: str) -> str:
    """Map a product description string to a category name.

    Args:
        description: Raw Description field from UCI Online Retail II.

    Returns:
        Category string, e.g. "KITCHEN_DINING" or "OTHER".
    """
    if not isinstance(description, str) or not description.strip():
        return FALLBACK_CATEGORY

    for cat, pattern in _COMPILED:
        if pattern.search(description):
            return cat

    return FALLBACK_CATEGORY


def encode_dataframe(df: pd.DataFrame, desc_col: str = "description") -> pd.DataFrame:
    """Add a 'category' column to the DataFrame by encoding the description column.

    Args:
        df:       DataFrame with at least one description column.
        desc_col: Name of the description column (default: 'description').

    Returns:
        DataFrame with new 'category' column appended (does NOT mutate input).
    """
    df = df.copy()
    df["category"] = df[desc_col].map(assign_category)
    return df


def get_category_distribution(df: pd.DataFrame, category_col: str = "category") -> pd.Series:
    """Return value counts for the category column (sorted descending)."""
    return df[category_col].value_counts()


def get_all_categories() -> list[str]:
    """Return all possible category names including OTHER."""
    return [cat for cat, _ in CATEGORY_RULES] + [FALLBACK_CATEGORY]


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("RED WOOLLY HOTTIE WHITE HEART.", "HOME_DECOR"),
        ("CREAM CUPID HEARTS COAT HANGER", "HOME_DECOR"),
        ("CHRISTMAS WREATH METAL SIGN",    "CHRISTMAS"),
        ("HAND WARMER BIRD DESIGN",        "GARDEN_OUTDOOR"),
        ("3 TIER CAKE TIN",                "KITCHEN_DINING"),
        ("GLASS STAR FROSTED T-LIGHT HOLDER", "CANDLES_LIGHTS"),
        ("STRAWBERRY CERAMIC TRINKET BOX", "BAGS_STORAGE"),
        ("GIRLS ALPHABET IRON ON PATCHES", "OTHER"),
    ]
    print(f"{'Description':<45} {'Expected':<20} {'Got':<20} {'OK'}")
    print("-" * 95)
    for desc, expected in test_cases:
        got = assign_category(desc)
        ok  = "✓" if got == expected else "✗"
        print(f"{desc:<45} {expected:<20} {got:<20} {ok}")
