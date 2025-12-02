import re
from typing import Optional, Union
import pandas as pd
import numpy as np
from .nicknames import NICKNAME_MAP


def normalize_phone(phone: str) -> str:
    """Normalize phone number by removing non-digits and handling country codes."""
    if not phone or not isinstance(phone, str):
        return ""
    
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]
    
    return digits


def normalize_phone_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized phone normalization for pandas Series."""
    digits = series.astype(str).str.replace(r'\D', '', regex=True)
    mask = (digits.str.len() == 11) & (digits.str[0] == '1')
    digits = digits.where(~mask, digits.str[1:])
    return digits


def normalize_address(address: str) -> str:
    """Normalize address by standardizing abbreviations and cleaning."""
    if not address or not isinstance(address, str):
        return ""
    
    address = address.lower().strip()
    
    abbreviations = {
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bav\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\brd\b': 'road',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bpkwy\b': 'parkway',
        r'\bapt\b': 'apartment',
        r'\bapts\b': 'apartments',
        r'\b#\b': '',
        r'\.': '',
        r',': '',
    }
    
    for pattern, replacement in abbreviations.items():
        address = re.sub(pattern, replacement, address)
    
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address


def normalize_address_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized address normalization for pandas Series."""
    result = series.astype(str).str.lower().str.strip()
    abbreviations = {
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bav\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\brd\b': 'road',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bpkwy\b': 'parkway',
        r'\bapt\b': 'apartment',
        r'\bapts\b': 'apartments',
        r'\b#\b': '',
        r'\.': '',
        r',': '',
    }
    for pattern, replacement in abbreviations.items():
        result = result.str.replace(pattern, replacement, regex=True)
    result = result.str.replace(r'\s+', ' ', regex=True).str.strip()
    return result


def normalize_name(name: str) -> str:
    """Normalize name by handling prefixes/suffixes, nicknames, and title case."""
    if not name or not isinstance(name, str):
        return ""
    
    name = name.strip()
    
    prefixes = ['mr', 'mrs', 'ms', 'dr', 'prof', 'rev']
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md']
    
    parts = name.lower().split()
    
    if parts and parts[0] in prefixes:
        parts = parts[1:]
    
    if len(parts) > 1 and parts[-1] in suffixes:
        parts = parts[:-1]
    
    expanded_parts = []
    for part in parts:
        expanded = NICKNAME_MAP.get(part, part)
        expanded_parts.append(expanded)
    
    normalized = ' '.join(expanded_parts)
    normalized = normalized.title()
    
    return normalized


def normalize_name_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized name normalization for pandas Series."""
    prefixes = {'mr', 'mrs', 'ms', 'dr', 'prof', 'rev'}
    suffixes = {'jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md'}
    
    result = series.astype(str).str.strip()
    
    # Vectorized processing using string operations
    # Split into words, process each word, then rejoin
    # Note: This is a simplified vectorized version - full nickname expansion
    # requires per-row processing but we optimize the common cases
    
    # Remove empty/nan values first
    mask = (result != '') & (result != 'nan') & result.notna()
    
    # For non-empty values, process vectorized
    processed = result.copy()
    if mask.any():
        # Lowercase and split (vectorized)
        lower_series = result[mask].str.lower()
        
        # Process each row - this still requires iteration but avoids function call overhead
        # by using list comprehension which is faster than .apply()
        processed_values = []
        for val in result:
            if not val or val == 'nan':
                processed_values.append("")
            else:
                parts = val.lower().split()
                if parts and parts[0] in prefixes:
                    parts = parts[1:]
                if len(parts) > 1 and parts[-1] in suffixes:
                    parts = parts[:-1]
                expanded_parts = [NICKNAME_MAP.get(part, part) for part in parts]
                normalized = ' '.join(expanded_parts)
                processed_values.append(normalized.title() if normalized else "")
        
        processed = pd.Series(processed_values, index=result.index)
    
    return processed


def normalize_email(email: str) -> str:
    """Normalize email by lowercasing and trimming."""
    if not email or not isinstance(email, str):
        return ""
    
    return email.lower().strip()


def normalize_email_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized email normalization for pandas Series."""
    return series.astype(str).str.lower().str.strip()


def normalize_string(text: str) -> str:
    """Generic string normalization: lowercase, trim, remove extra whitespace."""
    if not text or not isinstance(text, str):
        return ""
    
    return re.sub(r'\s+', ' ', text.lower().strip())


def normalize_string_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized string normalization for pandas Series."""
    return series.astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

