# validate_corpus.py  — run from big-data-project root
from pathlib import Path

EXPECTED = {
    "balzac":    ["le_pere_goriot", "eugenie_grandet", "la_cousine_bette"],
    "dumas":     ["les_trois_mousquetaires", "le_comte_de_monte_cristo"],
    "stendhal":  ["le_rouge_et_le_noir", "la_chartreuse_de_parme"],
    "hugo":      ["les_miserables", "notre_dame_de_paris"],
    "sand":      ["indiana"],
    "flaubert":  ["madame_bovary", "leducation_sentimentale"],
    "zola":      ["germinal", "nana", "lassommoir"],
    "dickens":   ["oliver_twist", "great_expectations", "bleak_house"],
}

RAW = Path("DATA/raw")
issues = []

for author, titles in EXPECTED.items():
    for title in titles:
        path = RAW / author / f"{title}.txt"
        
        # check exists + size
        if not path.exists():
            issues.append(f"MISSING  {path}")
            continue
        
        size_kb = path.stat().st_size // 1024
        
        # check it's not an HTML error page
        head = path.read_bytes()[:200]
        if b"<!DOCTYPE" in head or b"<html" in head:
            issues.append(f"HTML_ERR {path}  ({size_kb} KB)")
            continue
        
        # check UTF-8 decodable
        try:
            text = path.read_text(encoding="utf-8")
            words = len(text.split())
            print(f"  ok  {author:12s}  {title:35s}  {size_kb:5d} KB  {words:,} words")
        except UnicodeDecodeError:
            issues.append(f"ENCODING {path}")

print()
if issues:
    print("ISSUES FOUND:")
    for i in issues: print(" ", i)
else:
    print("All files clean.")