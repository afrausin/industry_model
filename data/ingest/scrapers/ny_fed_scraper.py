"""
Federal Reserve Bank of New York Data Scraper
==============================================
Downloads data from the NY Fed including:
- Survey of Consumer Expectations (SCE)
- Weekly Economic Index (WEI)
- Primary Dealer Survey
- Treasury market data

NY Fed Data: https://www.newyorkfed.org/research
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class NYFedScraper(BaseScraper):
    """
    Scraper for Federal Reserve Bank of New York data.
    
    Key datasets:
    - Survey of Consumer Expectations (inflation expectations)
    - Weekly Economic Index (high-frequency activity)
    - Liberty Street Economics blog insights
    """
    
    SOURCE_NAME = 'ny_fed'
    BASE_URL = 'https://www.newyorkfed.org'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'sce': '/microeconomics/sce',
        'sce_data': '/microeconomics/sce/downloads',
        'wei': '/research/policy/weekly-economic-index',
        'liberty_street': '/newsevents/news/research',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available NY Fed datasets."""
        return [
            {
                'name': 'Survey of Consumer Expectations',
                'key': 'sce',
                'description': 'Consumer inflation and economic expectations',
                'frequency': 'monthly'
            },
            {
                'name': 'SCE Inflation Expectations',
                'key': 'sce_inflation',
                'description': '1-year and 3-year ahead inflation expectations',
                'frequency': 'monthly'
            },
            {
                'name': 'Weekly Economic Index (WEI)',
                'key': 'wei',
                'description': 'High-frequency measure of real economic activity',
                'frequency': 'weekly'
            },
            {
                'name': 'Liberty Street Economics',
                'key': 'liberty_street',
                'description': 'Recent research blog posts from NY Fed economists',
                'frequency': 'ongoing'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific NY Fed dataset."""
        
        if dataset_key == 'sce':
            return self._download_sce(force)
        elif dataset_key == 'sce_inflation':
            return self._download_sce_inflation(force)
        elif dataset_key == 'wei':
            return self._download_wei(force)
        elif dataset_key == 'liberty_street':
            return self._download_liberty_street(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_sce(self, force: bool = False) -> Optional[Path]:
        """Download Survey of Consumer Expectations overview."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['sce']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract key statistics from the page
            page_text = soup.get_text(separator='\n', strip=True)
            
            # Look for data download links
            data_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv']) or 'download' in text:
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    data_links.append({
                        'url': full_url,
                        'text': link.get_text(strip=True)
                    })
            
            # Try to extract key metrics
            metrics = {}
            
            # Look for inflation expectations
            inflation_match = re.search(
                r'(?:median|mean)?\s*(?:one[- ]year|1[- ]year)\s*ahead\s*inflation[^\d]*(\d+\.?\d*)',
                page_text,
                re.IGNORECASE
            )
            if inflation_match:
                metrics['one_year_inflation_expectation'] = float(inflation_match.group(1))
            
            three_year_match = re.search(
                r'(?:median|mean)?\s*(?:three[- ]year|3[- ]year)\s*ahead\s*inflation[^\d]*(\d+\.?\d*)',
                page_text,
                re.IGNORECASE
            )
            if three_year_match:
                metrics['three_year_inflation_expectation'] = float(three_year_match.group(1))
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'metrics': metrics,
                'data_links': data_links,
                'page_content': page_text[:5000]
            }
            
            # Download available data files
            downloaded_files = []
            for dl in data_links[:3]:  # Limit downloads
                try:
                    if any(ext in dl['url'].lower() for ext in ['.xlsx', '.xls', '.csv']):
                        file_response = self.fetch_url(dl['url'])
                        filename = dl['url'].split('/')[-1].split('?')[0]
                        file_path = self.data_dir / filename
                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)
                        downloaded_files.append(filename)
                except Exception as e:
                    self.logger.warning(f"Failed to download {dl['url']}: {e}")
            
            data['downloaded_files'] = downloaded_files
            
            return self.save_json(
                data=data,
                filename='sce_overview.json',
                source_url=url,
                metadata={'metrics': metrics},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download SCE: {e}")
            return None
    
    def _download_sce_inflation(self, force: bool = False) -> Optional[Path]:
        """Download SCE inflation expectations data."""
        try:
            # Try to access the direct data download page
            url = f"{self.BASE_URL}{self.ENDPOINTS['sce_data']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find Excel/CSV downloads for inflation expectations
            inflation_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                if 'inflation' in text or 'expectations' in text:
                    if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv']):
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        inflation_links.append({
                            'url': full_url,
                            'text': link.get_text(strip=True)
                        })
            
            # Download available files
            downloaded_files = []
            for dl in inflation_links:
                try:
                    file_response = self.fetch_url(dl['url'])
                    filename = f"sce_inflation_{dl['url'].split('/')[-1].split('?')[0]}"
                    file_path = self.data_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    downloaded_files.append(filename)
                except Exception as e:
                    self.logger.warning(f"Failed to download inflation data: {e}")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'inflation_data_links': inflation_links,
                'downloaded_files': downloaded_files
            }
            
            return self.save_json(
                data=data,
                filename='sce_inflation.json',
                source_url=url,
                metadata={'file_count': len(downloaded_files)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download SCE inflation: {e}")
            return None
    
    def _download_wei(self, force: bool = False) -> Optional[Path]:
        """Download Weekly Economic Index."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['wei']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            page_text = soup.get_text(separator='\n', strip=True)
            
            # Extract current WEI value
            wei_value = None
            wei_match = re.search(
                r'(?:WEI|Weekly Economic Index)[^\d]*(-?\d+\.?\d*)',
                page_text,
                re.IGNORECASE
            )
            if wei_match:
                wei_value = float(wei_match.group(1))
            
            # Find data download links
            data_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv']):
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    data_links.append({
                        'url': full_url,
                        'text': link.get_text(strip=True)
                    })
            
            # Download data files
            downloaded_files = []
            for dl in data_links:
                try:
                    file_response = self.fetch_url(dl['url'])
                    filename = f"wei_{dl['url'].split('/')[-1].split('?')[0]}"
                    file_path = self.data_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    downloaded_files.append(filename)
                except Exception as e:
                    self.logger.warning(f"Failed to download WEI data: {e}")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'current_wei': wei_value,
                'data_links': data_links,
                'downloaded_files': downloaded_files,
                'page_content': page_text[:3000]
            }
            
            return self.save_json(
                data=data,
                filename='weekly_economic_index.json',
                source_url=url,
                metadata={'current_wei': wei_value},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download WEI: {e}")
            return None
    
    def _download_liberty_street(self, force: bool = False) -> Optional[Path]:
        """
        Download recent Liberty Street Economics blog posts WITH FULL CONTENT.
        
        Liberty Street Economics is the NY Fed's research blog where economists
        publish timely analysis on topics like:
        - Labor market dynamics
        - Inflation expectations
        - Financial conditions
        - Consumer behavior
        - Credit and lending
        
        This qualitative content provides insights that precede formal research papers.
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['liberty_street']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find blog post links
            posts = []
            for article in soup.find_all(['article', 'div'], class_=re.compile(r'post|article|entry')):
                title_elem = article.find(['h2', 'h3', 'a'])
                if title_elem:
                    link = title_elem.find('a') if title_elem.name != 'a' else title_elem
                    if link and link.get('href'):
                        href = link['href']
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        posts.append({
                            'title': link.get_text(strip=True),
                            'url': full_url
                        })
            
            # If no articles found, try simpler link extraction
            if not posts:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/libertystreeteconomics/' in href or '/research/' in href:
                        text = link.get_text(strip=True)
                        if len(text) > 20:  # Likely a post title
                            full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            posts.append({
                                'title': text,
                                'url': full_url
                            })
            
            # Deduplicate
            seen = set()
            unique_posts = []
            for p in posts:
                if p['url'] not in seen:
                    seen.add(p['url'])
                    unique_posts.append(p)
            
            # Download FULL CONTENT of recent posts
            downloaded_posts = []
            for post in unique_posts[:15]:  # Limit to recent 15 posts
                try:
                    post_response = self.fetch_url(post['url'])
                    post_soup = BeautifulSoup(post_response.text, 'html.parser')
                    
                    # Extract post date
                    post_date = None
                    date_elem = post_soup.find(['time', 'span'], class_=re.compile(r'date|time'))
                    if date_elem:
                        post_date = date_elem.get_text(strip=True)
                    
                    # Extract authors
                    authors = []
                    author_elems = post_soup.find_all(['a', 'span'], class_=re.compile(r'author'))
                    for a in author_elems:
                        author_text = a.get_text(strip=True)
                        if author_text and len(author_text) < 100:
                            authors.append(author_text)
                    
                    # Extract main content
                    content_div = post_soup.find(['article', 'div'], class_=re.compile(r'content|post-body|entry-content'))
                    if content_div:
                        content = content_div.get_text(separator='\n', strip=True)
                    else:
                        content = post_soup.get_text(separator='\n', strip=True)
                    
                    # Extract key topics/tags
                    tags = []
                    tag_elems = post_soup.find_all(['a', 'span'], class_=re.compile(r'tag|category|topic'))
                    for t in tag_elems:
                        tag_text = t.get_text(strip=True)
                        if tag_text and len(tag_text) < 50:
                            tags.append(tag_text)
                    
                    downloaded_posts.append({
                        'title': post['title'],
                        'url': post['url'],
                        'date': post_date,
                        'authors': list(set(authors))[:5],
                        'tags': list(set(tags))[:10],
                        'content': content[:10000],  # Limit content size
                        'word_count': len(content.split())
                    })
                    
                    self.logger.info(f"Downloaded Liberty Street post: {post['title'][:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch post content: {e}")
                    downloaded_posts.append({
                        'title': post['title'],
                        'url': post['url'],
                        'error': str(e)
                    })
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'post_count': len(downloaded_posts),
                'posts_with_content': len([p for p in downloaded_posts if 'content' in p]),
                'posts': downloaded_posts,
                'all_post_urls': [p['url'] for p in unique_posts]  # Reference to all found posts
            }
            
            return self.save_json(
                data=data,
                filename='liberty_street_posts.json',
                source_url=url,
                metadata={
                    'post_count': len(downloaded_posts),
                    'type': 'qualitative'
                },
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download Liberty Street: {e}")
            return None

