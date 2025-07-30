import re

def clean_and_structure_text(text: str):
    # Clean text
    cleaned = re.sub(r"[^\w\s,.:;\-()\[\]/]+", "", text)
    cleaned = re.sub(r"[\n\r]+", "\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    # Structure text (simple heuristics)
    structured = {
        "product_name": None,
        "ingredients": None,
        "features": None,
        "other_info": None
    }
    # Try to extract sections
    name_match = re.search(r"(?i)(product|name)[:\-]?\s*(.+)", cleaned)
    if name_match:
        structured["product_name"] = name_match.group(2).strip()
    ing_match = re.search(r"(?i)ingredients?[:\-]?\s*([^\n]+)", cleaned)
    if ing_match:
        structured["ingredients"] = ing_match.group(1).strip()
    # Features: look for 'features', 'details', etc.
    feat_match = re.search(r"(?i)(features?|details?)[:\-]?\s*([^\n]+)", cleaned)
    if feat_match:
        structured["features"] = feat_match.group(2).strip()
    # Other info: whatever is left
    structured["other_info"] = cleaned
    return cleaned, structured
