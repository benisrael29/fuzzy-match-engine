from typing import Union, Optional, Tuple
from datetime import datetime
from rapidfuzz import distance, fuzz
from dateutil import parser as date_parser
import re


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


def _parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Parse date string with multiple format attempts."""
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = str(date_str).strip()
    if not date_str or date_str.lower() in ('nan', 'none', 'null', ''):
        return None
    
    # Try dateutil parser first (handles most formats)
    try:
        return date_parser.parse(date_str, dayfirst=False, yearfirst=False)
    except (ValueError, TypeError):
        pass
    
    # Try common explicit formats
    common_formats = [
        '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y',
        '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
        '%Y.%m.%d', '%m.%d.%Y', '%d.%m.%Y',
        '%Y%m%d', '%m%d%Y', '%d%m%Y',
        '%B %d, %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y',
        '%d-%b-%Y', '%Y-%b-%d', '%b-%d-%Y',
        '%d/%b/%Y', '%Y/%b/%d', '%b/%d/%Y',
    ]
    
    for fmt in common_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # Try parsing with dayfirst=True (for DD-MM-YYYY formats)
    try:
        return date_parser.parse(date_str, dayfirst=True)
    except (ValueError, TypeError):
        pass
    
    # Try parsing with yearfirst=True (for YYYY-MM-DD formats)
    try:
        return date_parser.parse(date_str, yearfirst=True)
    except (ValueError, TypeError):
        pass
    
    return None


def _extract_date_components(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Extract year, month, day from date string using regex patterns."""
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    
    # Pattern for YYYY-MM-DD or YYYY/MM/DD
    pattern1 = r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'
    match = re.search(pattern1, date_str)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                # Validate it's a real date
                datetime(year, month, day)
                return (year, month, day)
            except (ValueError, TypeError):
                pass
    
    # Pattern for ambiguous MM-DD-YYYY or DD-MM-YYYY
    # Try both interpretations and pick the one that makes sense
    pattern2 = r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})'
    match = re.search(pattern2, date_str)
    if match:
        val1, val2, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        # Try MM-DD-YYYY interpretation
        if 1 <= val1 <= 12 and 1 <= val2 <= 31:
            try:
                datetime(year, val1, val2)
                return (year, val1, val2)
            except (ValueError, TypeError):
                pass
        
        # Try DD-MM-YYYY interpretation
        if 1 <= val2 <= 12 and 1 <= val1 <= 31:
            try:
                datetime(year, val2, val1)
                return (year, val2, val1)
            except (ValueError, TypeError):
                pass
        
        # If neither works, prefer MM-DD-YYYY if first value could be month
        if 1 <= val1 <= 12:
            return (year, val1, val2)
    
    # Pattern for YYYYMMDD
    pattern3 = r'(\d{4})(\d{2})(\d{2})'
    match = re.search(pattern3, date_str)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                datetime(year, month, day)
                return (year, month, day)
            except (ValueError, TypeError):
                pass
    
    return None


def _compare_date_components(comp1: Optional[Tuple[int, int, int]], 
                            comp2: Optional[Tuple[int, int, int]]) -> float:
    """Compare date components and return similarity score."""
    if comp1 is None or comp2 is None:
        return 0.0
    
    year1, month1, day1 = comp1
    year2, month2, day2 = comp2
    
    # Exact match
    if year1 == year2 and month1 == month2 and day1 == day2:
        return 1.0
    
    # Calculate component-wise similarity
    year_match = 1.0 if year1 == year2 else 0.0
    month_match = 1.0 if month1 == month2 else 0.0
    day_match = 1.0 if day1 == day2 else 0.0
    
    # Weighted component matching
    component_score = (year_match * 0.5 + month_match * 0.3 + day_match * 0.2)
    
    # Also consider temporal distance for partial matches
    try:
        dt1 = datetime(year1, month1, day1)
        dt2 = datetime(year2, month2, day2)
        days_diff = abs((dt1 - dt2).days)
        
        # Temporal similarity (decays with days difference)
        temporal_score = 1.0 / (1.0 + days_diff / 365.0)
        
        # Combine component matching with temporal similarity
        return max(component_score, temporal_score * 0.7)
    except (ValueError, TypeError):
        # Invalid date (e.g., Feb 30), return component score only
        return component_score * 0.5


def date_similarity(date1: Union[str, datetime], date2: Union[str, datetime]) -> float:
    """
    Calculate date similarity handling multiple formats and time components.
    
    Handles:
    - Different date formats (MM-DD-YYYY, DD-MM-YYYY, YYYY-MM-DD, etc.)
    - Different separators (-, /, .)
    - Time components (if present, compares dates only)
    - Partial dates (compares available components)
    
    Returns similarity score (0-1 scale).
    """
    # Convert to strings if datetime objects
    if isinstance(date1, datetime):
        date1_str = date1.strftime('%Y-%m-%d')
    else:
        date1_str = str(date1) if date1 is not None else ''
    
    if isinstance(date2, datetime):
        date2_str = date2.strftime('%Y-%m-%d')
    else:
        date2_str = str(date2) if date2 is not None else ''
    
    # Handle empty/null values
    if not date1_str or not date2_str:
        return 0.0
    
    date1_str = date1_str.strip()
    date2_str = date2_str.strip()
    
    if not date1_str or not date2_str:
        return 0.0
    
    # Try parsing both dates
    dt1 = _parse_date_flexible(date1_str)
    dt2 = _parse_date_flexible(date2_str)
    
    # If both parsed successfully, use temporal comparison
    if dt1 is not None and dt2 is not None:
        # Extract date components (ignore time)
        date_only1 = dt1.date()
        date_only2 = dt2.date()
        
        if date_only1 == date_only2:
            return 1.0
        
        days_diff = abs((date_only1 - date_only2).days)
        
        # Same year: more forgiving
        if dt1.year == dt2.year:
            return 1.0 / (1.0 + days_diff / 365.0)
        else:
            year_diff = abs(dt1.year - dt2.year)
            return 1.0 / (1.0 + days_diff / 365.0 + year_diff * 0.5)
    
    # If parsing failed, try component extraction
    comp1 = _extract_date_components(date1_str)
    comp2 = _extract_date_components(date2_str)
    
    if comp1 is not None and comp2 is not None:
        return _compare_date_components(comp1, comp2)
    
    # If one parsed and other didn't, try to extract components from the parsed one
    if dt1 is not None and comp2 is not None:
        comp1 = (dt1.year, dt1.month, dt1.day)
        return _compare_date_components(comp1, comp2)
    
    if dt2 is not None and comp1 is not None:
        comp2 = (dt2.year, dt2.month, dt2.day)
        return _compare_date_components(comp1, comp2)
    
    # If both parsing and component extraction failed, try string similarity
    # (for cases like "11-12-2000" vs "11/12/2000")
    normalized1 = re.sub(r'[-/.\s]', '', date1_str)
    normalized2 = re.sub(r'[-/.\s]', '', date2_str)
    
    if normalized1 == normalized2:
        return 1.0
    
    # Use Levenshtein similarity as fallback
    return levenshtein_similarity(normalized1, normalized2) * 0.5

