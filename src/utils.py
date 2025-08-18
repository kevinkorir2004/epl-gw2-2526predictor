
import re

TEAM_CLEAN_RE = re.compile(r"\s*(?:FC|AFC)\.?$", re.IGNORECASE)

def clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    replacements = {
        "Manchester Utd": "Manchester United",
        "Man United": "Manchester United",
        "Man Utd": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Newcastle Utd": "Newcastle United",
        "West Brom": "West Bromwich Albion",
        "Bournemouth": "AFC Bournemouth",
        "Brighton": "Brighton & Hove Albion",
        "Leeds Utd": "Leeds United",
        "Nott'm Forest": "Nottingham Forest",
        "Nott Forest": "Nottingham Forest",
        "Sheffield Utd": "Sheffield United",
        "Ipswich": "Ipswich Town",
        "Sunderland": "Sunderland AFC",
    }
    name = TEAM_CLEAN_RE.sub("", name)
    return replacements.get(name, name)

def outcome_to_label(ftr: str) -> int:
    mapping = {"H": 0, "D": 1, "A": 2}
    return mapping.get(str(ftr).strip().upper(), None)

def label_to_outcome(label: int) -> str:
    mapping = {0: "H", 1: "D", 2: "A"}
    return mapping.get(int(label), "?")
