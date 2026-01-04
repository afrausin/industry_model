"""
Base Scraper Framework with Download Tracking
==============================================
Provides base functionality for all data scrapers including:
- Download tracking (what has been downloaded and when)
- Rate limiting to respect API/server limits
- Error handling and retry logic
- Consistent file saving patterns
- Version archiving (keeps historical versions)
"""

import json
import hashlib
import logging
import shutil
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DownloadTracker:
    """
    Tracks all downloads across scrapers.
    Maintains a JSON file with download history including:
    - Source URL
    - Download timestamp
    - File hash (for detecting changes)
    - Local file path
    - Status (success/failed)
    """
    
    def __init__(self, tracker_path: Path):
        self.tracker_path = tracker_path
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_tracker()
    
    def _load_tracker(self):
        """Load existing tracker data or initialize empty tracker."""
        if self.tracker_path.exists():
            with open(self.tracker_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'downloads': {}
            }
            self._save_tracker()
    
    def _save_tracker(self):
        """Persist tracker data to disk."""
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.tracker_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content for change detection."""
        return hashlib.sha256(content).hexdigest()
    
    def is_downloaded(self, source_key: str) -> bool:
        """Check if a source has been downloaded before."""
        return source_key in self.data['downloads']
    
    def get_download_info(self, source_key: str) -> Optional[dict]:
        """Get download info for a source key."""
        return self.data['downloads'].get(source_key)
    
    def has_changed(self, source_key: str, new_hash: str) -> bool:
        """Check if content has changed since last download."""
        info = self.get_download_info(source_key)
        if info is None:
            return True
        return info.get('content_hash') != new_hash
    
    def record_download(
        self,
        source_key: str,
        source_url: str,
        local_path: Path,
        content_hash: str,
        status: str = 'success',
        metadata: Optional[dict] = None
    ):
        """Record a download in the tracker."""
        self.data['downloads'][source_key] = {
            'source_url': source_url,
            'local_path': str(local_path),
            'content_hash': content_hash,
            'downloaded_at': datetime.now().isoformat(),
            'status': status,
            'metadata': metadata or {}
        }
        self._save_tracker()
    
    def get_all_downloads(self, source_prefix: Optional[str] = None) -> dict:
        """Get all downloads, optionally filtered by source prefix."""
        if source_prefix is None:
            return self.data['downloads']
        return {
            k: v for k, v in self.data['downloads'].items()
            if k.startswith(source_prefix)
        }
    
    def get_summary(self) -> dict:
        """Get a summary of all downloads by source."""
        summary = {}
        for key in self.data['downloads']:
            source = key.split('/')[0] if '/' in key else key
            summary[source] = summary.get(source, 0) + 1
        return {
            'total_downloads': len(self.data['downloads']),
            'by_source': summary,
            'last_updated': self.data['last_updated']
        }


class BaseScraper(ABC):
    """
    Abstract base class for all data scrapers.
    
    Provides common functionality:
    - HTTP session management with retry logic
    - Rate limiting
    - Download tracking integration
    - Consistent error handling
    """
    
    # Class-level configuration
    SOURCE_NAME = 'base'
    BASE_URL = ''
    RATE_LIMIT_SECONDS = 1.0  # Minimum time between requests
    MAX_RETRIES = 3
    RETRY_DELAY = 5.0
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        self.data_dir = data_dir / self.SOURCE_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.data_dir / 'archive'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_request_time = 0
        self._session = None
    
    @property
    def session(self) -> requests.Session:
        """Lazy-initialized HTTP session with browser-like headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            })
        return self._session
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()
    
    def fetch_url(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None
    ) -> requests.Response:
        """
        Fetch URL with rate limiting and retry logic.
        
        Args:
            url: URL to fetch
            params: Optional query parameters
            headers: Optional additional headers
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        self._rate_limit()
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    raise
    
    def _archive_existing_file(self, file_path: Path) -> Optional[Path]:
        """
        Archive an existing file before overwriting.
        
        Creates a timestamped copy in the archive directory.
        Format: archive/filename_YYYYMMDD_HHMMSS.ext
        
        Returns:
            Path to archived file, or None if file didn't exist
        """
        if not file_path.exists():
            return None
        
        # Get the file's modification time for the archive timestamp
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        timestamp = mtime.strftime('%Y%m%d_%H%M%S')
        
        # Create archive filename
        stem = file_path.stem
        suffix = file_path.suffix
        archive_name = f"{stem}_{timestamp}{suffix}"
        archive_path = self.archive_dir / archive_name
        
        # Don't archive if this exact version already exists
        if archive_path.exists():
            return archive_path
        
        # Copy to archive
        shutil.copy2(file_path, archive_path)
        self.logger.debug(f"Archived {file_path.name} → archive/{archive_name}")
        
        return archive_path
    
    def save_file(
        self,
        content: bytes,
        filename: str,
        source_url: str,
        source_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        force: bool = False
    ) -> Optional[Path]:
        """
        Save content to file and record in tracker.
        
        Automatically archives existing file before overwriting.
        
        Args:
            content: Raw bytes to save
            filename: Name of file to save
            source_url: Original source URL
            source_key: Unique key for tracking (defaults to source/filename)
            metadata: Additional metadata to store
            force: Force download even if unchanged
            
        Returns:
            Path to saved file, or None if skipped (unchanged)
        """
        if source_key is None:
            source_key = f"{self.SOURCE_NAME}/{filename}"
        
        content_hash = self.tracker.compute_hash(content)
        
        # Check if content has changed
        if not force and not self.tracker.has_changed(source_key, content_hash):
            self.logger.info(f"Skipping {filename} (unchanged)")
            return None
        
        file_path = self.data_dir / filename
        
        # Archive existing file before overwriting
        self._archive_existing_file(file_path)
        
        # Save new file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Record in tracker
        self.tracker.record_download(
            source_key=source_key,
            source_url=source_url,
            local_path=file_path,
            content_hash=content_hash,
            metadata=metadata
        )
        
        self.logger.info(f"Saved {filename}")
        return file_path
    
    def save_json(
        self,
        data: dict,
        filename: str,
        source_url: str,
        source_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        force: bool = False
    ) -> Optional[Path]:
        """
        Save JSON data to file.
        
        For change detection, excludes 'downloaded_at' timestamp so that
        identical data doesn't trigger unnecessary saves.
        """
        # Create a copy without timestamp for hash comparison
        data_for_hash = {k: v for k, v in data.items() if k != 'downloaded_at'}
        hash_content = json.dumps(data_for_hash, indent=2, default=str, sort_keys=True).encode('utf-8')
        
        # Full content includes timestamp
        content = json.dumps(data, indent=2, default=str).encode('utf-8')
        
        if source_key is None:
            source_key = f"{self.SOURCE_NAME}/{filename}"
        
        content_hash = self.tracker.compute_hash(hash_content)
        
        # Check if content has changed (excluding timestamp)
        if not force and not self.tracker.has_changed(source_key, content_hash):
            self.logger.info(f"Skipping {filename} (unchanged)")
            return None
        
        file_path = self.data_dir / filename
        
        # Archive existing file before overwriting
        self._archive_existing_file(file_path)
        
        # Save new file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Record in tracker
        self.tracker.record_download(
            source_key=source_key,
            source_url=source_url,
            local_path=file_path,
            content_hash=content_hash,
            metadata=metadata
        )
        
        self.logger.info(f"Saved {filename}")
        return file_path
    
    def load_existing_json(self, filename: str) -> Optional[dict]:
        """
        Load existing JSON data from a file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Parsed JSON data, or None if file doesn't exist
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load {filename}: {e}")
            return None
    
    def get_existing_keys(self, filename: str, items_key: str, id_field: str) -> set:
        """
        Get set of existing item IDs from a JSON file.
        
        Useful for incremental updates - only download items not in this set.
        
        Args:
            filename: JSON file to check
            items_key: Key containing the list of items (e.g., 'statements', 'speeches')
            id_field: Field to use as unique ID (e.g., 'date', 'url')
            
        Returns:
            Set of existing IDs
        """
        data = self.load_existing_json(filename)
        if data is None:
            return set()
        
        items = data.get(items_key, [])
        return {item.get(id_field) for item in items if item.get(id_field)}
    
    def merge_items(
        self,
        existing_data: Optional[dict],
        new_items: list,
        items_key: str,
        id_field: str,
        sort_field: Optional[str] = None,
        reverse_sort: bool = True
    ) -> dict:
        """
        Merge new items with existing data, avoiding duplicates.
        
        Args:
            existing_data: Existing JSON data (or None)
            new_items: List of new items to add
            items_key: Key for the items list in the data structure
            id_field: Field to use for deduplication
            sort_field: Optional field to sort by after merging
            reverse_sort: Sort in descending order (newest first)
            
        Returns:
            Merged data structure
        """
        if existing_data is None:
            existing_data = {items_key: []}
        
        existing_items = existing_data.get(items_key, [])
        existing_ids = {item.get(id_field) for item in existing_items}
        
        # Add only new items
        added_count = 0
        for item in new_items:
            item_id = item.get(id_field)
            if item_id and item_id not in existing_ids:
                existing_items.append(item)
                existing_ids.add(item_id)
                added_count += 1
        
        # Sort if requested
        if sort_field:
            existing_items.sort(key=lambda x: x.get(sort_field, ''), reverse=reverse_sort)
        
        existing_data[items_key] = existing_items
        
        if added_count > 0:
            self.logger.info(f"Added {added_count} new items (total: {len(existing_items)})")
        
        return existing_data
    
    @abstractmethod
    def get_available_datasets(self) -> list[dict]:
        """
        Return list of available datasets for this source.
        
        Each dict should contain:
        - name: Human-readable name
        - key: Unique identifier
        - description: Brief description
        - frequency: Update frequency (daily, weekly, monthly, quarterly)
        """
        pass
    
    @abstractmethod
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """
        Download a specific dataset.
        
        Args:
            dataset_key: Key from get_available_datasets()
            force: Force download even if unchanged
            
        Returns:
            Path to downloaded file(s), or None if skipped
        """
        pass
    
    def download_all(self, force: bool = False) -> dict[str, Optional[Path]]:
        """Download all available datasets."""
        results = {}
        for dataset in self.get_available_datasets():
            key = dataset['key']
            try:
                results[key] = self.download_dataset(key, force=force)
            except Exception as e:
                self.logger.error(f"Failed to download {key}: {e}")
                results[key] = None
        return results
    
    def get_archive_versions(self, filename: str) -> list[dict]:
        """
        Get list of archived versions for a file.
        
        Returns list of dicts with:
        - path: Path to archived file
        - timestamp: datetime of archive
        - size_bytes: File size
        """
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        
        versions = []
        for archive_file in self.archive_dir.glob(f"{stem}_*{suffix}"):
            # Parse timestamp from filename
            try:
                ts_str = archive_file.stem.replace(f"{stem}_", "")
                ts = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                versions.append({
                    'path': archive_file,
                    'timestamp': ts,
                    'size_bytes': archive_file.stat().st_size
                })
            except ValueError:
                continue
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions
    
    def cleanup_old_archives(self, filename: str, keep_last: int = 10) -> int:
        """
        Remove old archive versions, keeping only the most recent ones.
        
        Args:
            filename: Base filename to clean up
            keep_last: Number of recent versions to keep
            
        Returns:
            Number of files deleted
        """
        versions = self.get_archive_versions(filename)
        
        deleted = 0
        for version in versions[keep_last:]:
            version['path'].unlink()
            deleted += 1
            self.logger.info(f"Deleted old archive: {version['path'].name}")
        
        return deleted
    
    def close(self):
        """Clean up resources."""
        if self._session is not None:
            self._session.close()
            self._session = None

