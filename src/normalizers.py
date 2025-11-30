import re
from typing import Optional
from .nicknames import NICKNAME_MAP


def normalize_phone(phone: str) -> str:
    """Normalize phone number by removing non-digits and handling country codes."""
    if not phone or not isinstance(phone, str):
        return ""
    
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]
    
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


def normalize_email(email: str) -> str:
    """Normalize email by lowercasing and trimming."""
    if not email or not isinstance(email, str):
        return ""
    
    return email.lower().strip()


def normalize_string(text: str) -> str:
    """Generic string normalization: lowercase, trim, remove extra whitespace."""
    if not text or not isinstance(text, str):
        return ""
    
    return re.sub(r'\s+', ' ', text.lower().strip())

