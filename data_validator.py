"""
Data validation and input sanitization module for ASTINA.
Provides comprehensive data quality checks and input cleaning.
"""
import pandas as pd
import numpy as np
import re
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger("graphnet.data_validator")

class DataSanitizer:
    """Sanitize and clean user input data"""
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """
        Sanitize string input by removing potentially harmful characters.
        
        Args:
            text: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove null bytes and control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized.strip()
    
    @staticmethod
    def sanitize_column_name(name: str) -> str:
        """
        Sanitize column names to be safe for database and processing.
        
        Args:
            name: Column name to sanitize
            
        Returns:
            Sanitized column name
        """
        if not isinstance(name, str):
            name = str(name)
        
        # Remove special characters except alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'col_' + sanitized
            
        return sanitized.lower() if sanitized else 'unnamed'
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize entire dataframe by cleaning column names and string values.
        
        Args:
            df: Input dataframe
            
        Returns:
            Sanitized dataframe
        """
        df_sanitized = df.copy()
        
        # Sanitize column names
        original_cols = df_sanitized.columns.tolist()
        sanitized_cols = [DataSanitizer.sanitize_column_name(col) for col in original_cols]
        
        # Create mapping to handle duplicates
        col_mapping = {}
        final_cols = []
        for i, (orig, san) in enumerate(zip(original_cols, sanitized_cols)):
            if san not in col_mapping.values():
                col_mapping[orig] = san
                final_cols.append(san)
            else:
                # Add suffix for duplicates
                new_name = f"{san}_{i}"
                col_mapping[orig] = new_name
                final_cols.append(new_name)
        
        df_sanitized.columns = final_cols
        
        # Sanitize string columns
        for col in df_sanitized.select_dtypes(include=['object']).columns:
            df_sanitized[col] = df_sanitized[col].apply(
                lambda x: DataSanitizer.sanitize_string(str(x)) if pd.notna(x) else x
            )
        
        return df_sanitized

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def check_basic_integrity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic data integrity checks.
        
        Args:
            df: Dataframe to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check if dataframe is empty
        if df.empty:
            results['is_valid'] = False
            results['issues'].append("Dataframe kosong - tidak ada data untuk diproses")
            return results
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            results['warnings'].append(f"Kolom sepenuhnya kosong: {empty_cols[:5]}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Terdapat {duplicate_count} baris duplikat ({duplicate_count/len(df)*100:.1f}%)")
        
        # Check for columns with single unique value (constant columns)
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        if constant_cols:
            results['warnings'].append(f"Kolom dengan nilai konstan (tidak informatif): {constant_cols[:5]}")
        
        # Statistics
        results['stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': int(duplicate_count),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        return results
    
    @staticmethod
    def check_column_types(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate and analyze column types.
        
        Args:
            df: Dataframe to analyze
            
        Returns:
            Dictionary with column type analysis
        """
        results = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'mixed_type_columns': []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                results['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                results['datetime_columns'].append(col)
            elif df[col].nunique() / len(df) < 0.1:  # Low cardinality
                results['categorical_columns'].append(col)
            elif df[col].dtype == 'object':
                # Check if it's text or categorical
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 50:
                    results['text_columns'].append(col)
                else:
                    results['categorical_columns'].append(col)
        
        return results
    
    @staticmethod
    def check_data_ranges(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if numeric values are within reasonable ranges.
        
        Args:
            df: Dataframe to check
            
        Returns:
            Dictionary with range validation results
        """
        results = {
            'outliers': [],
            'suspicious_values': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Check for extreme values using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(df) * 100
                    if outlier_pct > 5:  # More than 5% outliers is suspicious
                        results['outliers'].append({
                            'column': col,
                            'count': len(outliers),
                            'percentage': outlier_pct
                        })
                
                # Check for negative values in columns that shouldn't have them
                if any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'age', 'count']):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        results['suspicious_values'].append({
                            'column': col,
                            'issue': 'negative_values',
                            'count': negative_count
                        })
                        
            except Exception as e:
                logger.warning(f"Error checking ranges for column {col}: {e}")
        
        return results
    
    @staticmethod
    def validate_for_ml(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate dataframe specifically for machine learning processing.
        
        Args:
            df: Dataframe to validate
            target_col: Name of target column (if supervised learning)
            
        Returns:
            Dictionary with ML-specific validation results
        """
        results = {
            'is_ready': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check minimum data size
        if len(df) < 100:
            results['warnings'].append("Dataset sangat kecil (<100 baris). Hasil ML mungkin tidak reliable.")
        
        # Check for target column if specified
        if target_col and target_col in df.columns:
            unique_values = df[target_col].nunique()
            if unique_values < 2:
                results['issues'].append(f"Target column '{target_col}' hanya memiliki {unique_values} nilai unik.")
                results['is_ready'] = False
            elif unique_values > 50:
                results['warnings'].append(f"Target column '{target_col}' memiliki {unique_values} nilai unik. Pertimbangkan untuk klasifikasi multi-class atau regression.")
        
        # Check for high cardinality categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            cardinality = df[col].nunique()
            if cardinality > len(df) * 0.5:
                results['recommendations'].append(
                    f"Kolom '{col}' memiliki kardinalitas tinggi ({cardinality} nilai unik). "
                    "Pertimbangkan encoding yang lebih efisien atau drop kolom."
                )
        
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        if high_missing_cols:
            results['recommendations'].append(
                f"Kolom dengan missing values >50%: {high_missing_cols[:5]}. "
                "Pertimbangkan untuk drop kolom tersebut."
            )
        
        return results

def comprehensive_validation(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform comprehensive data validation.
    
    Args:
        df: Dataframe to validate
        target_col: Optional target column name
        
    Returns:
        Tuple of (is_valid, validation_results)
    """
    all_results = {
        'basic_integrity': {},
        'column_types': {},
        'data_ranges': {},
        'ml_readiness': {}
    }
    
    try:
        # Basic integrity check
        all_results['basic_integrity'] = DataValidator.check_basic_integrity(df)
        
        # Column type analysis
        all_results['column_types'] = DataValidator.check_column_types(df)
        
        # Data range validation
        all_results['data_ranges'] = DataValidator.check_data_ranges(df)
        
        # ML readiness check
        all_results['ml_readiness'] = DataValidator.validate_for_ml(df, target_col)
        
        # Overall validity
        is_valid = (
            all_results['basic_integrity']['is_valid'] and
            all_results['ml_readiness']['is_ready']
        )
        
        return is_valid, all_results
        
    except Exception as e:
        logger.error(f"Error during comprehensive validation: {e}")
        return False, {'error': str(e)}

def display_validation_results(results: Dict[str, Any]) -> None:
    """
    Display validation results in a user-friendly format.
    
    Args:
        results: Validation results dictionary
    """
    if 'error' in results:
        if STREAMLIT_AVAILABLE and st:
            st.error(f"❌ Error saat validasi: {results['error']}")
        else:
            print(f"❌ Error saat validasi: {results['error']}")
        return
    
    if STREAMLIT_AVAILABLE and st:
        st.subheader("📊 Hasil Validasi Data")
        
        # Basic integrity
        basic = results.get('basic_integrity', {})
        if basic:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Baris", f"{basic['stats']['total_rows']:,}")
            with col2:
                st.metric("Total Kolom", basic['stats']['total_columns'])
            with col3:
                st.metric("Missing Values", f"{basic['stats']['missing_values']:,}")
            with col4:
                st.metric("Memory Usage", f"{basic['stats']['memory_usage_mb']:.1f} MB")
            
            if basic['issues']:
                st.error("❌ **Masalah Ditemukan:**")
                for issue in basic['issues']:
                    st.markdown(f"- {issue}")
            
            if basic['warnings']:
                st.warning("⚠️ **Peringatan:**")
                for warning in basic['warnings']:
                    st.markdown(f"- {warning}")
        
        # ML readiness
        ml_ready = results.get('ml_readiness', {})
        if ml_ready:
            if ml_ready['issues']:
                st.error("❌ **Masalah ML:**")
                for issue in ml_ready['issues']:
                    st.markdown(f"- {issue}")
            
            if ml_ready['warnings']:
                st.warning("⚠️ **Peringatan ML:**")
                for warning in ml_ready['warnings']:
                    st.markdown(f"- {warning}")
            
            if ml_ready['recommendations']:
                st.info("💡 **Rekomendasi:**")
                for rec in ml_ready['recommendations']:
                    st.markdown(f"- {rec}")
    else:
        # Fallback for non-streamlit environments
        print("📊 Hasil Validasi Data")
        
        basic = results.get('basic_integrity', {})
        if basic:
            print(f"Total Baris: {basic['stats']['total_rows']:,}")
            print(f"Total Kolom: {basic['stats']['total_columns']}")
            print(f"Missing Values: {basic['stats']['missing_values']:,}")
            print(f"Memory Usage: {basic['stats']['memory_usage_mb']:.1f} MB")
            
            if basic['issues']:
                print("❌ Masalah Ditemukan:")
                for issue in basic['issues']:
                    print(f"- {issue}")
            
            if basic['warnings']:
                print("⚠️ Peringatan:")
                for warning in basic['warnings']:
                    print(f"- {warning}")
        
        ml_ready = results.get('ml_readiness', {})
        if ml_ready:
            if ml_ready['issues']:
                print("❌ Masalah ML:")
                for issue in ml_ready['issues']:
                    print(f"- {issue}")
            
            if ml_ready['warnings']:
                print("⚠️ Peringatan ML:")
                for warning in ml_ready['warnings']:
                    print(f"- {warning}")
            
            if ml_ready['recommendations']:
                print("💡 Rekomendasi:")
                for rec in ml_ready['recommendations']:
                    print(f"- {rec}")
