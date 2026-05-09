"""
fetch_corpus.py  —  reproducible corpus acquisition from Project Gutenberg
Run once: python fetch_corpus.py
"""

import time
import urllib.request
from pathlib import Path

CORPUS = [
    # (author_slug, title_slug, gutenberg_id)
    # --- Romantic (FR) ---
    ("balzac",   "le_pere_goriot",            1237),
    ("balzac",   "eugenie_grandet",           1419),
    ("balzac",   "la_cousine_bette",          1327),
    ("dumas",    "les_trois_mousquetaires",   1257),
    ("dumas",    "le_comte_de_monte_cristo",  1184),
    ("stendhal", "le_rouge_et_le_noir",       798),
    ("stendhal", "la_chartreuse_de_parme",    799),
    ("hugo",     "les_miserables",            135),
    ("hugo",     "notre_dame_de_paris",       2610),
    ("sand",     "indiana",                   5838),
    # --- Realist (FR) ---
    ("flaubert", "madame_bovary",             2413),
    ("flaubert", "leducation_sentimentale",   2179),
    ("zola",     "germinal",                  4225),
    ("zola",     "nana",                      5250),
    ("zola",     "lassommoir",                4910),
    # --- Realist (EN) ---
    ("dickens",  "oliver_twist",              730),
    ("dickens",  "great_expectations",        1400),
    ("dickens",  "bleak_house",               1023),
]

RAW_DIR = Path("DATA/raw")
BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
FALLBACK  = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch(author: str, title: str, gid: int) -> None:
    dest = RAW_DIR / author / f"{title}.txt"
    if dest.exists():
        print(f"  skip  {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for url_tmpl in (BASE_URL, FALLBACK):
        url = url_tmpl.format(id=gid)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  ok    {dest}  ({dest.stat().st_size // 1024} KB)")
            time.sleep(1)          # be polite to Gutenberg
            return
        except Exception:
            continue

    print(f"  FAIL  {author}/{title}  (id={gid}) — check URL manually")


if __name__ == "__main__":
    for author, title, gid in CORPUS:
        fetch(author, title, gid)
    print("Done.")