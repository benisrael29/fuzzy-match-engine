from typing import Union
from datetime import datetime
from rapidfuzz import distance, fuzz
from dateutil import parser as date_parser


def levenshtein_similarity(str1: str, str2: str) -> float:
    """Calculate Levenshtein similarity using rapidfuzz (0-1 scale)."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    return distance.Levenshtein.normalized_similarity(str1, str2)


def jaro_winkler_similarity(str1: str, str2: str) -> float:
    """Calculate Jaro-Winkler similarity using rapidfuzz (0-1 scale)."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    return distance.JaroWinkler.normalized_similarity(str1, str2)


def token_set_ratio(str1: str, str2: str) -> float:
    """Calculate token set ratio using rapidfuzz (0-1 scale)."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    return fuzz.token_set_ratio(str1, str2) / 100.0


def token_sort_ratio(str1: str, str2: str) -> float:
    """Calculate token sort ratio using rapidfuzz (0-1 scale)."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    return fuzz.token_sort_ratio(str1, str2) / 100.0


def numeric_similarity(num1: Union[int, float], num2: Union[int, float]) -> float:
    """Calculate numeric similarity using ratio-based approach (0-1 scale)."""
    try:
        num1 = float(num1)
        num2 = float(num2)
    except (ValueError, TypeError):
        return 0.0
    
    if num1 == num2:
        return 1.0
    
    max_val = max(abs(num1), abs(num2), 1.0)
    diff = abs(num1 - num2)
    
    return 1.0 - (diff / max_val)


def date_similarity(date1: Union[str, datetime], date2: Union[str, datetime]) -> float:
    """Calculate date similarity using temporal distance (0-1 scale)."""
    try:
        if isinstance(date1, str):
            date1 = date_parser.parse(date1)
        if isinstance(date2, str):
            date2 = date_parser.parse(date2)
    except (ValueError, TypeError):
        return 0.0
    
    if not isinstance(date1, datetime) or not isinstance(date2, datetime):
        return 0.0
    
    if date1 == date2:
        return 1.0
    
    days_diff = abs((date1 - date2).days)
    
    if date1.year == date2.year:
        return 1.0 / (1.0 + days_diff / 365.0)
    else:
        year_diff = abs(date1.year - date2.year)
        return 1.0 / (1.0 + days_diff / 365.0 + year_diff * 0.5)

