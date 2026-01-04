"""
Peterson Institute for International Economics (PIIE) Scraper
==============================================================
Downloads trade and international economics research including:
- Trade policy analysis (tariffs, trade wars)
- Global economic outlook
- Policy briefs

PIIE: https://www.piie.com/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class PIIEScraper(BaseScraper):
    """
    Scraper for Peterson Institute for International Economics.
    
    PIIE is the premier think tank for international trade and
    economic policy. Critical for tariff and trade war analysis.
    """
    
    SOURCE_NAME = 'piie'
    BASE_URL = 'https://www.piie.com'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'research': '/research',
        'trade': '/topics/trade-and-investment-policy',
        'blogs': '/blogs/realtime-economics',
        'charts': '/research/piie-charts',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available PIIE datasets."""
        return [
            {
                'name': 'Trade Policy Research',
                'key': 'trade_research',
                'description': 'Analysis of tariffs, trade agreements, and trade wars',
                'frequency': 'ongoing',
                'type': 'qualitative'
            },
            {
                'name': 'PIIE Charts',
                'key': 'charts',
                'description': 'Data visualizations on global economics',
                'frequency': 'ongoing',
                'type': 'quantitative'
            },
            {
                'name': 'RealTime Economics Blog',
                'key': 'blog',
                'description': 'Commentary on current economic events',
                'frequency': 'daily',
                'type': 'qualitative'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific PIIE dataset."""
        if dataset_key == 'trade_research':
            return self._download_trade_research(force)
        elif dataset_key == 'charts':
            return self._download_charts(force)
        elif dataset_key == 'blog':
            return self._download_blog(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_trade_research(self, force: bool = False) -> Optional[Path]:
        """Download trade policy research papers incrementally."""
        try:
            # Get existing paper URLs
            existing_urls = self.get_existing_keys('trade_research.json', 'papers', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing papers, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['trade']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            papers = []
            
            # Find research items
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(view|item|result|research)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title'))
                if not title_elem:
                    title_elem = item.find(['h2', 'h3', 'h4'])
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    date_elem = item.find(class_=re.compile(r'date'))
                    date = date_elem.get_text(strip=True) if date_elem else None
                    
                    author_elem = item.find(class_=re.compile(r'author'))
                    authors = author_elem.get_text(strip=True) if author_elem else None
                    
                    type_elem = item.find(class_=re.compile(r'type|category'))
                    paper_type = type_elem.get_text(strip=True) if type_elem else None
                    
                    if title and len(title) > 10:
                        papers.append({
                            'title': title,
                            'url': link,
                            'date': date,
                            'authors': authors,
                            'type': paper_type
                        })
            
            # Deduplicate
            seen = set()
            unique_papers = []
            for p in papers:
                if p['title'] not in seen:
                    seen.add(p['title'])
                    unique_papers.append(p)
            
            # Filter to new papers only
            if not force and existing_urls:
                new_papers = [p for p in unique_papers if p.get('url') not in existing_urls]
            else:
                new_papers = unique_papers
            
            if not new_papers and existing_urls:
                self.logger.info("No new PIIE trade research to download")
                return self.data_dir / 'trade_research.json'
            
            # Merge with existing
            existing_data = self.load_existing_json('trade_research.json')
            merged_data = self.merge_items(existing_data, new_papers, 'papers', 'url')
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['paper_count'] = len(merged_data.get('papers', []))
            
            return self.save_json(
                data=merged_data,
                filename='trade_research.json',
                source_url=url,
                metadata={'paper_count': merged_data['paper_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download trade research: {e}")
            return None
    
    def _download_charts(self, force: bool = False) -> Optional[Path]:
        """Download PIIE charts data incrementally."""
        try:
            existing_urls = self.get_existing_keys('charts.json', 'charts', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing charts, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['charts']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            charts = []
            
            for item in soup.find_all(['article', 'div'], class_=re.compile(r'(chart|view|item)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    desc_elem = item.find(['p', 'div'], class_=re.compile(r'(description|summary)'))
                    description = desc_elem.get_text(strip=True) if desc_elem else None
                    
                    if title and len(title) > 5:
                        charts.append({
                            'title': title,
                            'url': link,
                            'description': description[:200] if description else None
                        })
            
            seen = set()
            unique_charts = []
            for c in charts:
                if c['title'] not in seen:
                    seen.add(c['title'])
                    unique_charts.append(c)
            
            if not force and existing_urls:
                new_charts = [c for c in unique_charts if c.get('url') not in existing_urls]
            else:
                new_charts = unique_charts
            
            if not new_charts and existing_urls:
                self.logger.info("No new PIIE charts to download")
                return self.data_dir / 'charts.json'
            
            existing_data = self.load_existing_json('charts.json')
            merged_data = self.merge_items(existing_data, new_charts, 'charts', 'url')
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['chart_count'] = len(merged_data.get('charts', []))
            
            return self.save_json(
                data=merged_data,
                filename='charts.json',
                source_url=url,
                metadata={'chart_count': merged_data['chart_count'], 'type': 'quantitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download charts: {e}")
            return None
    
    def _download_blog(self, force: bool = False) -> Optional[Path]:
        """Download RealTime Economics blog posts incrementally."""
        try:
            existing_urls = self.get_existing_keys('blog.json', 'posts', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing posts, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['blogs']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            posts = []
            
            for item in soup.find_all(['article', 'div'], class_=re.compile(r'(blog|post|view|item)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    date_elem = item.find(class_=re.compile(r'date'))
                    date = date_elem.get_text(strip=True) if date_elem else None
                    
                    author_elem = item.find(class_=re.compile(r'author'))
                    author = author_elem.get_text(strip=True) if author_elem else None
                    
                    excerpt_elem = item.find(['p', 'div'], class_=re.compile(r'(excerpt|summary|teaser)'))
                    excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else None
                    
                    if title and len(title) > 10:
                        posts.append({
                            'title': title,
                            'url': link,
                            'date': date,
                            'author': author,
                            'excerpt': excerpt[:300] if excerpt else None
                        })
            
            seen = set()
            unique_posts = []
            for p in posts:
                if p['title'] not in seen:
                    seen.add(p['title'])
                    unique_posts.append(p)
            
            if not force and existing_urls:
                new_posts = [p for p in unique_posts if p.get('url') not in existing_urls]
            else:
                new_posts = unique_posts
            
            if not new_posts and existing_urls:
                self.logger.info("No new PIIE blog posts to download")
                return self.data_dir / 'blog.json'
            
            existing_data = self.load_existing_json('blog.json')
            merged_data = self.merge_items(existing_data, new_posts, 'posts', 'url')
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['post_count'] = len(merged_data.get('posts', []))
            
            return self.save_json(
                data=merged_data,
                filename='blog.json',
                source_url=url,
                metadata={'post_count': merged_data['post_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download blog: {e}")
            return None

