import pandas as pd
import os
import json
import hashlib
import streamlit as st
import gc
import time

CACHE_DIR = os.environ.get("CACHE_DIR", "cache")

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_file_hash(file_or_name, file_size=None, sample_size=4096):
    """
    Generate a stable hash for cache lookup.
    Uses file metadata plus small head/tail samples to avoid false cache hits
    from different files that happen to share the same name and size.
    """
    if hasattr(file_or_name, 'read') and hasattr(file_or_name, 'seek'):
        uploaded_file = file_or_name
        current_pos = 0

        try:
            current_pos = uploaded_file.tell()
        except Exception:
            current_pos = 0

        try:
            uploaded_file.seek(0)
            head = uploaded_file.read(sample_size) or b''

            tail = b''
            total_size = getattr(uploaded_file, 'size', file_size) or 0
            if total_size > sample_size:
                uploaded_file.seek(max(total_size - sample_size, 0))
                tail = uploaded_file.read(sample_size) or b''

            uploaded_file.seek(current_pos)
        except Exception:
            head = b''
            tail = b''

        digest = hashlib.md5()
        digest.update(str(getattr(uploaded_file, 'name', '')).encode('utf-8', errors='ignore'))
        digest.update(str(total_size).encode('utf-8'))
        digest.update(head if isinstance(head, (bytes, bytearray)) else bytes(head))
        digest.update(tail if isinstance(tail, (bytes, bytearray)) else bytes(tail))
        return digest.hexdigest()

    hash_str = f"{file_or_name}_{file_size}"
    return hashlib.md5(hash_str.encode()).hexdigest()

def save_to_cache(df, file_hash, final_features, metadata):
    """Save processed dataframe and its metadata to disk"""
    ensure_cache_dir()
    
    parquet_path = os.path.join(CACHE_DIR, f"{file_hash}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    
    try:
        # Save dataframe using parquet with LZ4 compression (faster than snappy)
        df.to_parquet(parquet_path, compression='lz4', index=False)
        
        # Save metadata
        meta_payload = {
            'final_features': final_features,
            'preprocessing_metadata': metadata
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_payload, f, indent=4)
            
        print(f"Cache saved: {file_hash}")
        return True
    except Exception as e:
        st.warning(f"Gagal menyimpan cache: {str(e)}")
        return False

def load_from_cache(file_hash):
    """Load processed dataframe and metadata from disk if available"""
    parquet_path = os.path.join(CACHE_DIR, f"{file_hash}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    
    if os.path.exists(parquet_path) and os.path.exists(meta_path):
        try:
            # Load metadata first
            with open(meta_path, 'r') as f:
                meta_payload = json.load(f)
                
            # Load dataframe
            df = pd.read_parquet(parquet_path)
            
            print(f"Cache loaded: {file_hash}")
            return df, meta_payload['final_features'], meta_payload['preprocessing_metadata']
        except Exception as e:
            st.error(f"Gagal memuat cache: {str(e)}")
            return None, None, None
            
    return None, None, None

def get_cache_path(file_hash):
    """Return the parquet file path and metadata without loading the dataframe into memory"""
    parquet_path = os.path.join(CACHE_DIR, f"{file_hash}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    
    if os.path.exists(parquet_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta_payload = json.load(f)
            return parquet_path, meta_payload['final_features'], meta_payload['preprocessing_metadata']
        except Exception as e:
            st.error(f"Gagal memuat metadata cache: {str(e)}")
            return None, None, None
            
    return None, None, None

def clear_old_cache(max_files=10):
    """Optionally clear old cache files to save space"""
    if not os.path.exists(CACHE_DIR):
        return
        
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)]
    if len(files) > max_files * 2: # Each cache has 2 files (.parquet and .json)
        # Sort by modification time
        files.sort(key=os.path.getmtime)
        # Delete oldest half
        for f in files[:len(files)//2]:
            try:
                os.remove(f)
            except:
                pass

def smart_cache_eviction(max_size_mb=500, max_files=20):
    """
    Smart cache eviction based on LRU + size + last accessed time.
    
    Args:
        max_size_mb: Maximum total cache size in MB
        max_files: Maximum number of cache files to keep
    """
    if not os.path.exists(CACHE_DIR):
        return
    
    try:
        cache_files = []
        total_size = 0
        
        # Collect cache file metadata
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    file_mtime = stat.st_mtime
                    file_atime = stat.st_atime
                    
                    cache_files.append({
                        'path': file_path,
                        'size': file_size,
                        'mtime': file_mtime,
                        'atime': file_atime,
                        'name': filename
                    })
                    total_size += file_size
                except Exception as e:
                    print(f"Error getting stats for {file_path}: {e}")
                    continue
        
        # Check if eviction is needed
        total_size_mb = total_size / (1024 * 1024)
        num_files = len(cache_files)
        
        if total_size_mb <= max_size_mb and num_files <= max_files:
            return  # No eviction needed
        
        # Calculate eviction score for each file
        # Lower score = higher priority to evict
        for file_info in cache_files:
            # Score based on: recency (higher is better), size (smaller is better)
            age_hours = (time.time() - file_info['atime']) / 3600
            size_mb = file_info['size'] / (1024 * 1024)
            
            # Recency score (0-1, higher is better)
            recency_score = max(0, 1 - age_hours / 168)  # Decay over 1 week
            
            # Size score (0-1, smaller is better)
            size_score = max(0, 1 - size_mb / 100)  # Penalize files > 100MB
            
            # Combined score (higher is better, keep these)
            file_info['score'] = (recency_score * 0.7) + (size_score * 0.3)
        
        # Sort by score (lowest first = evict first)
        cache_files.sort(key=lambda x: x['score'])
        
        # Evict files until constraints are met
        files_evicted = 0
        for file_info in cache_files:
            if total_size_mb <= max_size_mb and num_files <= max_files:
                break
            
            try:
                os.remove(file_info['path'])
                total_size -= file_info['size']
                total_size_mb = total_size / (1024 * 1024)
                num_files -= 1
                files_evicted += 1
                print(f"Evicted cache file: {file_info['name']} ({file_info['size']/1024/1024:.2f}MB)")
            except Exception as e:
                print(f"Error evicting {file_info['path']}: {e}")
        
        if files_evicted > 0:
            print(f"Smart cache eviction completed: {files_evicted} files removed")
    
    except Exception as e:
        print(f"Error during smart cache eviction: {e}")

def get_cache_stats():
    """
    Get cache statistics for monitoring.
    
    Returns:
        Dictionary with cache statistics
    """
    if not os.path.exists(CACHE_DIR):
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'oldest_file': None,
            'newest_file': None
        }
    
    try:
        files = []
        total_size = 0
        oldest_time = float('inf')
        newest_time = 0
        oldest_file = None
        newest_file = None
        
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                file_size = stat.st_size
                file_mtime = stat.st_mtime
                
                total_size += file_size
                files.append(filename)
                
                if file_mtime < oldest_time:
                    oldest_time = file_mtime
                    oldest_file = filename
                
                if file_mtime > newest_time:
                    newest_time = file_mtime
                    newest_file = filename
        
        return {
            'total_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_file': oldest_file,
            'newest_file': newest_file,
            'oldest_age_hours': (time.time() - oldest_time) / 3600 if oldest_time != float('inf') else None,
            'newest_age_hours': (time.time() - newest_time) / 3600 if newest_time != 0 else None
        }
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'oldest_file': None,
            'newest_file': None
        }
