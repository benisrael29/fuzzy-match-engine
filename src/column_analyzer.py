import re
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dateutil import parser as date_parser
from .algorithms import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    numeric_similarity,
    date_similarity,
    token_set_ratio
)


COLUMN_TYPES = {
    'string_name': 'string_name',
    'string_general': 'string_general',
    'numeric': 'numeric',
    'date': 'date',
    'email': 'email',
    'phone': 'phone'
}

ALGORITHM_MAP = {
    'string_name': jaro_winkler_similarity,
    'string_general': levenshtein_similarity,
    'numeric': numeric_similarity,
    'date': date_similarity,
    'email': token_set_ratio,
    'phone': token_set_ratio
}


def detect_column_type(series: pd.Series, column_name: str = "") -> str:
    """
    Detect column type using pandas dtypes and custom heuristics.
    
    Returns:
        Column type string (string_name, string_general, numeric, date, email, phone)
    """
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    
    if series.dtype != 'object':
        return 'string_general'
    
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'string_general'
    
    sample_size = min(100, len(non_null))
    sample = non_null.head(sample_size).astype(str)
    
    if _is_email_column(sample):
        return 'email'
    
    if _is_phone_column(sample):
        return 'phone'
    
    if _is_date_column(sample):
        return 'date'
    
    if _is_name_column(sample, column_name):
        return 'string_name'
    
    return 'string_general'


def _is_email_column(sample: pd.Series) -> bool:
    """Check if column contains email addresses."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    email_count = sum(sample.str.match(email_pattern, na=False))
    return email_count / len(sample) > 0.5


def _is_phone_column(sample: pd.Series) -> bool:
    """Check if column contains phone numbers."""
    digit_count = sample.str.count(r'\d')
    avg_digits = digit_count.mean()
    
    phone_pattern = r'[\d\s\-\(\)\+]{10,}'
    phone_count = sum(sample.str.match(phone_pattern, na=False))
    
    return (phone_count / len(sample) > 0.5) or (avg_digits >= 7)


def _is_date_column(sample: pd.Series) -> bool:
    """Check if column contains dates."""
    date_count = 0
    for value in sample:
        try:
            date_parser.parse(str(value))
            date_count += 1
        except (ValueError, TypeError):
            pass
    
    return date_count / len(sample) > 0.5


def _is_name_column(sample: pd.Series, column_name: str) -> bool:
    """Check if column contains names."""
    name_keywords = ['name', 'first', 'last', 'full', 'person', 'contact']
    column_lower = column_name.lower()
    
    if any(keyword in column_lower for keyword in name_keywords):
        return True
    
    title_case_count = sum(sample.str.istitle())
    if title_case_count / len(sample) > 0.6:
        return True
    
    common_name_words = ['john', 'mary', 'robert', 'james', 'michael', 'william',
                        'david', 'richard', 'joseph', 'thomas', 'charles', 'daniel']
    sample_lower = sample.str.lower()
    name_word_count = sum(sample_lower.str.contains('|'.join(common_name_words), na=False))
    
    return name_word_count / len(sample) > 0.3


def select_algorithm(column_type: str):
    """
    Select optimal algorithm for column type.
    
    Returns:
        Algorithm function
    """
    return ALGORITHM_MAP.get(column_type, levenshtein_similarity)


def analyze_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column_mappings: Optional[list] = None
) -> Dict[Tuple[str, str], Dict]:
    """
    Analyze column pairs and return algorithm selections.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        column_mappings: Optional list of dicts with 'source1' and 'source2' keys
    
    Returns:
        Dict mapping (col1, col2) tuples to analysis results
    """
    results = {}
    
    if column_mappings:
        for mapping in column_mappings:
            col1 = mapping.get('source1')
            col2 = mapping.get('source2')
            
            if col1 not in df1.columns:
                raise ValueError(f"Column '{col1}' not found in source1")
            if col2 not in df2.columns:
                raise ValueError(f"Column '{col2}' not found in source2")
            
            type1 = detect_column_type(df1[col1], col1)
            type2 = detect_column_type(df2[col2], col2)
            
            column_type = type1 if type1 == type2 else 'string_general'
            algorithm = select_algorithm(column_type)
            
            results[(col1, col2)] = {
                'type1': type1,
                'type2': type2,
                'column_type': column_type,
                'algorithm': algorithm,
                'weight': mapping.get('weight', 1.0)
            }
    else:
        for col1 in df1.columns:
            if col1 in df2.columns:
                type1 = detect_column_type(df1[col1], col1)
                type2 = detect_column_type(df2[col1], col1)
                
                column_type = type1 if type1 == type2 else 'string_general'
                algorithm = select_algorithm(column_type)
                
                results[(col1, col1)] = {
                    'type1': type1,
                    'type2': type2,
                    'column_type': column_type,
                    'algorithm': algorithm,
                    'weight': 1.0
                }
    
    return results


def auto_detect_column_mappings(
    query_record: Dict[str, Any],
    master_df: pd.DataFrame
) -> list:
    """
    Auto-detect column mappings between a query record and master dataset.
    
    Args:
        query_record: Dictionary with query fields
        master_df: Master dataset DataFrame
    
    Returns:
        List of column mapping dicts with 'source1' and 'source2' keys
    """
    mappings = []
    master_cols_lower = {col.lower(): col for col in master_df.columns}
    
    for query_key, query_value in query_record.items():
        if query_value is None or (isinstance(query_value, str) and query_value.strip() == ''):
            continue
        
        query_key_lower = query_key.lower()
        
        best_match = None
        best_score = 0.0
        
        for master_col_lower, master_col in master_cols_lower.items():
            if query_key_lower == master_col_lower:
                best_match = master_col
                best_score = 1.0
                break
            
            if query_key_lower in master_col_lower or master_col_lower in query_key_lower:
                score = len(set(query_key_lower) & set(master_col_lower)) / max(len(query_key_lower), len(master_col_lower))
                if score > best_score:
                    best_score = score
                    best_match = master_col
        
        if best_match and best_score > 0.3:
            mappings.append({
                'source1': query_key,
                'source2': best_match,
                'weight': 1.0
            })
    
    return mappings


def analyze_query_columns(
    query_record: Dict[str, Any],
    master_df: pd.DataFrame,
    column_mappings: Optional[list] = None
) -> Dict[Tuple[str, str], Dict]:
    """
    Analyze column pairs between a query record and master dataset.
    
    Args:
        query_record: Dictionary with query fields
        master_df: Master dataset DataFrame
        column_mappings: Optional list of dicts with 'source1' and 'source2' keys
    
    Returns:
        Dict mapping (col1, col2) tuples to analysis results
    """
    query_df = pd.DataFrame([query_record])
    
    if column_mappings is None:
        column_mappings = auto_detect_column_mappings(query_record, master_df)
    
    if not column_mappings:
        return {}
    
    results = {}
    
    for mapping in column_mappings:
        col1 = mapping.get('source1')
        col2 = mapping.get('source2')
        
        if col1 not in query_df.columns:
            continue
        if col2 not in master_df.columns:
            continue
        
        type1 = detect_column_type(query_df[col1], col1)
        type2 = detect_column_type(master_df[col2], col2)
        
        column_type = type1 if type1 == type2 else 'string_general'
        algorithm = select_algorithm(column_type)
        
        results[(col1, col2)] = {
            'type1': type1,
            'type2': type2,
            'column_type': column_type,
            'algorithm': algorithm,
            'weight': mapping.get('weight', 1.0)
        }
    
    return results
