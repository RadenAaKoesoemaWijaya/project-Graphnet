"""
ENHANCED ENCODING RECOMMENDATIONS FOR ASTINA
============================================

This module provides advanced encoding and preprocessing strategies for the ASTINA platform.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import streamlit as st

def get_strategy_name(cardinality, target_column=None):
    """Returns the strategy name based on cardinality and target availability"""
    if cardinality <= 5:
        return 'one_hot'
    elif cardinality <= 20:
        return 'binary'
    elif cardinality <= 100:
        return 'frequency_rare'
    else:
        return 'target' if target_column is not None else 'frequency'

def enhanced_categorical_encoding(df, categorical_columns, target_column=None):
    """Enhanced categorical encoding with multiple strategies"""
    
    import category_encoders as ce
    
    encoded_features = []
    encoding_metadata = {}
    
    for col in categorical_columns:
        cardinality = df[col].nunique()
        
        # Strategy selection based on cardinality
        if cardinality <= 5:
            # One-Hot Encoding for low cardinality
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded = encoder.fit_transform(df[[col]])
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            
        elif cardinality <= 20:
            # Binary Encoding for medium cardinality
            encoder = ce.BinaryEncoder()
            encoded = encoder.fit_transform(df[col])
            feature_names = [f"{col}_binary_{i}" for i in range(encoded.shape[1])]
            
        elif cardinality <= 100:
            # Frequency Encoding + Rare Category Grouping
            freq_map = df[col].value_counts()
            rare_threshold = max(5, len(df) * 0.01)  # 1% or minimum 5
            
            # Group rare categories
            rare_categories = freq_map[freq_map < rare_threshold].index
            df[col + '_processed'] = df[col].where(~df[col].isin(rare_categories), 'RARE')
            
            # Frequency encoding
            freq_encoding = df[col + '_processed'].map(freq_map)
            encoded = freq_encoding.values.reshape(-1, 1)
            feature_names = [f"{col}_freq_encoded"]
            
        else:
            # Target Encoding for high cardinality (with proper validation)
            if target_column is not None:
                encoder = ce.TargetEncoder(smoothing=10)
                encoded = encoder.fit_transform(df[col], df[target_column])
                feature_names = [f"{col}_target_encoded"]
            else:
                # Fallback to frequency encoding
                freq_map = df[col].value_counts()
                encoded = df[col].map(freq_map).values.reshape(-1, 1)
                feature_names = [f"{col}_freq_encoded"]
        
        # Store encoder for future use
        encoding_metadata[col] = {
            'encoder': encoder,
            'strategy': get_strategy_name(cardinality, target_column),
            'feature_names': feature_names,
            'cardinality': cardinality
        }
        
        encoded_features.append(encoded)
    
    return np.hstack(encoded_features), encoding_metadata

def advanced_missing_handling(df):
    """Advanced missing value handling with indicators"""
    
    df_processed = df.copy()
    missing_indicators = {}
    
    # Create missing indicators
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_indicators[f"{col}_missing"] = df[col].isnull().astype(int)
    
    # Add missing indicators to dataframe
    if missing_indicators:
        df_indicators = pd.DataFrame(missing_indicators, index=df.index)
        df_processed = pd.concat([df_processed, df_indicators], axis=1)
    
    # Strategy selection based on missing rate and data type
    for col in df.columns:
        missing_rate = df[col].isnull().sum() / len(df)
        
        if missing_rate == 0:
            continue
        elif missing_rate < 0.05:
            # Simple imputation for low missing rate
            if df[col].dtype in ['float64', 'int64']:
                df_processed[col] = df[col].fillna(df[col].median())
            else:
                df_processed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                
        elif missing_rate < 0.20:
            # KNN imputation for moderate missing rate
            if df[col].dtype in ['float64', 'int64']:
                imputer = KNNImputer(n_neighbors=5)
                df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                
        elif missing_rate < 0.40:
            # Iterative imputation for high missing rate
            if df[col].dtype in ['float64', 'int64']:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                
        else:
            # Drop column if missing rate is too high
            df_processed.drop(col, axis=1, inplace=True)
    
    return df_processed, missing_indicators

def advanced_feature_scaling(df, numerical_columns):
    """Advanced feature scaling with outlier handling"""
    
    df_scaled = df.copy()
    scaling_metadata = {}
    
    for col in numerical_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # Detect outliers
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
        outlier_rate = outliers / len(data)
        
        # Strategy selection
        if outlier_rate > 0.1:  # More than 10% outliers
            # Robust scaling
            scaler = RobustScaler()
            strategy = 'robust'
            
        elif abs(data.skew()) > 2:  # Highly skewed
            # Power transformation + standard scaling
            scaler = PowerTransformer(method='yeo-johnson')
            strategy = 'power_transform'
            
        else:
            # Standard scaling
            scaler = StandardScaler()
            strategy = 'standard'
        
        # Apply scaling
        df_scaled[f"{col}_scaled"] = scaler.fit_transform(df[[col]]).ravel()
        
        # Store metadata
        scaling_metadata[col] = {
            'scaler': scaler,
            'strategy': strategy,
            'outlier_rate': outlier_rate,
            'skewness': data.skew()
        }
    
    return df_scaled, scaling_metadata

class EncoderManager:
    """Manages encoder persistence and consistency"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.metadata = {}
    
    def fit_encoders(self, df, target_column=None):
        """Fit all encoders and scalers"""
        
        # Categorical encoding
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            encoded_features, encoding_metadata = enhanced_categorical_encoding(
                df, categorical_columns, target_column
            )
            self.encoders['categorical'] = encoding_metadata
        
        # Numerical scaling
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            df_scaled, scaling_metadata = advanced_feature_scaling(df, numerical_columns)
            self.scalers['numerical'] = scaling_metadata
        
        # Store metadata
        self.metadata = {
            'fit_date': datetime.now(),
            'original_shape': df.shape,
            'categorical_columns': categorical_columns.tolist(),
            'numerical_columns': numerical_columns.tolist()
        }
    
    def transform_new_data(self, df):
        """Transform new data using fitted encoders"""
        
        try:
            df_transformed = df.copy()
            
            # Apply categorical encoding
            if 'categorical' in self.encoders:
                for col, metadata in self.encoders['categorical'].items():
                    if col in df.columns:
                        encoder = metadata['encoder']
                        if hasattr(encoder, 'transform'):
                            encoded = encoder.transform(df[[col]])
                            feature_names = metadata['feature_names']
                            df_transformed[feature_names] = encoded
            
            # Apply numerical scaling
            if 'numerical' in self.scalers:
                for col, metadata in self.scalers['numerical'].items():
                    if col in df.columns:
                        scaler = metadata['scaler']
                        df_transformed[f"{col}_scaled"] = scaler.transform(df[[col]]).ravel()
            
            return df_transformed
            
        except Exception as e:
            raise ValueError(f"Error transforming new data: {str(e)}")
    
    def save_encoders(self, filepath):
        """Save encoders to file"""
        joblib.dump({
            'encoders': self.encoders,
            'scalers': self.scalers,
            'metadata': self.metadata
        }, filepath)
    
    def load_encoders(self, filepath):
        """Load encoders from file"""
        data = joblib.load(filepath)
        self.encoders = data['encoders']
        self.scalers = data['scalers']
        self.metadata = data['metadata']

def validate_data_quality(df, reference_metadata=None):
    """Validate data quality against reference or general standards"""
    
    validation_results = {
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check for missing values
    missing_analysis = df.isnull().sum()
    high_missing_cols = missing_analysis[missing_analysis > len(df) * 0.3].index.tolist()
    
    if high_missing_cols:
        validation_results['issues'].append(
            f"High missing values in columns: {high_missing_cols}"
        )
        validation_results['recommendations'].append(
            "Consider imputation or remove high-missing columns"
        )
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        validation_results['issues'].append(f"Duplicate columns found: {duplicate_cols}")
    
    # Check cardinality changes (if reference provided)
    if reference_metadata:
        for col, ref_cardinality in reference_metadata.get('cardinality', {}).items():
            if col in df.columns:
                current_cardinality = df[col].nunique()
                if current_cardinality > ref_cardinality * 1.5:
                    validation_results['warnings'].append(
                        f"High cardinality increase in {col}: {ref_cardinality} -> {current_cardinality}"
                    )
    
    return validation_results

def enhanced_feature_engineering(df):
    """Placeholder for enhanced feature engineering"""
    # In a real scenario, this would include domain-specific features
    return df

def enhanced_preprocess_insurance_claims(df, encoder_manager=None, target_column=None):
    """Enhanced preprocessing with all improvements"""
    
    # Initialize encoder manager if not provided
    if encoder_manager is None:
        encoder_manager = EncoderManager()
    
    # Data quality validation
    validation_results = validate_data_quality(df)
    
    if validation_results['issues']:
        st.warning("Data quality issues detected:")
        for issue in validation_results['issues']:
            st.write(f"⚠️ {issue}")
    
    # Advanced missing value handling
    df_processed, missing_indicators = advanced_missing_handling(df)
    
    # Enhanced categorical encoding
    categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
    encoding_metadata = {}
    if len(categorical_columns) > 0:
        encoded_features, encoding_metadata = enhanced_categorical_encoding(
            df_processed, categorical_columns, target_column
        )
    
    # Feature scaling
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns
    scaling_metadata = {}
    if len(numerical_columns) > 0:
        df_scaled, scaling_metadata = advanced_feature_scaling(df_processed, numerical_columns)
    else:
        df_scaled = df_processed
    
    # Feature engineering
    df_final = enhanced_feature_engineering(df_scaled)
    
    # Store all metadata
    preprocessing_metadata = {
        'validation_results': validation_results,
        'missing_indicators': missing_indicators,
        'encoding_metadata': encoding_metadata,
        'scaling_metadata': scaling_metadata,
        'encoder_manager': encoder_manager
    }
    
    return df_final, df_final.select_dtypes(include=[np.number]).columns.tolist(), preprocessing_metadata
