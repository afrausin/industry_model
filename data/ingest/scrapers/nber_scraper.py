"""
National Bureau of Economic Research (NBER) Scraper
====================================================
Downloads research from NBER including:
- Business cycle dates (recession/expansion)
- Working papers
- Research digests

NBER: https://www.nber.org/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class NBERScraper(BaseScraper):
    """
    Scraper for National Bureau of Economic Research data.
    
    NBER is the official arbiter of US business cycles (recessions).
    Also publishes influential working papers in economics.
    """
    
    SOURCE_NAME = 'nber'
    BASE_URL = 'https://www.nber.org'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'cycles': '/research/data/us-business-cycle-expansions-and-contractions',
        'working_papers': '/papers',
        'digest': '/digest',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available NBER datasets."""
        return [
            {
                'name': 'Business Cycle Dates',
                'key': 'business_cycles',
                'description': 'Official US recession and expansion dates',
                'frequency': 'event-driven',
                'type': 'reference'
            },
            {
                'name': 'Working Papers (Recent)',
                'key': 'working_papers',
                'description': 'Recent NBER working papers - frontier economic research',
                'frequency': 'weekly',
                'type': 'qualitative'
            },
            {
                'name': 'NBER Digest',
                'key': 'digest',
                'description': 'Summaries of recent NBER research',
                'frequency': 'monthly',
                'type': 'qualitative'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific NBER dataset."""
        if dataset_key == 'business_cycles':
            return self._download_business_cycles(force)
        elif dataset_key == 'working_papers':
            return self._download_working_papers(force)
        elif dataset_key == 'digest':
            return self._download_digest(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_business_cycles(self, force: bool = False) -> Optional[Path]:
        """
        Download official US business cycle dates.
        
        The NBER Business Cycle Dating Committee determines the 
        peaks and troughs of the US business cycle.
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['cycles']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the business cycle table
            cycles = []
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                headers = []
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    if cells:
                        cell_text = [c.get_text(strip=True) for c in cells]
                        if not headers and any('peak' in c.lower() or 'trough' in c.lower() for c in cell_text):
                            headers = cell_text
                        elif headers:
                            if len(cell_text) >= 2:
                                cycles.append({
                                    'peak': cell_text[0] if len(cell_text) > 0 else None,
                                    'trough': cell_text[1] if len(cell_text) > 1 else None,
                                    'contraction_months': cell_text[2] if len(cell_text) > 2 else None,
                                    'expansion_months': cell_text[3] if len(cell_text) > 3 else None,
                                })
            
            # Also extract any text about current cycle status
            content = soup.get_text(separator='\n', strip=True)
            
            # Look for current status
            current_status = None
            if 'current' in content.lower():
                # Try to find sentences about current economic status
                sentences = content.split('.')
                for s in sentences:
                    if 'current' in s.lower() and ('expansion' in s.lower() or 'recession' in s.lower()):
                        current_status = s.strip()
                        break
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'title': 'US Business Cycle Expansions and Contractions',
                'description': 'Official NBER business cycle dates',
                'current_status': current_status,
                'cycles': cycles,
                'cycle_count': len(cycles)
            }
            
            return self.save_json(
                data=data,
                filename='business_cycles.json',
                source_url=url,
                metadata={'cycle_count': len(cycles), 'type': 'reference'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download business cycles: {e}")
            return None
    
    def _download_working_papers(self, force: bool = False) -> Optional[Path]:
        """Download recent NBER working papers incrementally."""
        try:
            # Get existing paper numbers to avoid re-downloading
            existing_papers = self.get_existing_keys('working_papers.json', 'papers', 'paper_number')
            if existing_papers and not force:
                self.logger.info(f"Have {len(existing_papers)} existing papers, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['working_papers']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            papers = []
            
            # Find paper listings
            for article in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(paper|result|item)')):
                title_elem = article.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title'))
                if not title_elem:
                    title_elem = article.find(['h2', 'h3', 'h4'])
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    # Get paper link
                    link = None
                    link_elem = article.find('a', href=re.compile(r'/papers/w\d+'))
                    if link_elem:
                        link = link_elem['href']
                        if not link.startswith('http'):
                            link = f"{self.BASE_URL}{link}"
                    
                    # Get authors
                    author_elem = article.find(class_=re.compile(r'author'))
                    authors = author_elem.get_text(strip=True) if author_elem else None
                    
                    # Get paper number
                    paper_num = None
                    if link:
                        match = re.search(r'w(\d+)', link)
                        if match:
                            paper_num = f"w{match.group(1)}"
                    
                    # Get abstract/summary
                    abstract_elem = article.find(class_=re.compile(r'(abstract|summary|excerpt)'))
                    abstract = abstract_elem.get_text(strip=True) if abstract_elem else None
                    
                    if title and len(title) > 10:
                        papers.append({
                            'paper_number': paper_num,
                            'title': title,
                            'authors': authors,
                            'url': link,
                            'abstract': abstract[:500] if abstract else None
                        })
            
            # Deduplicate by paper number or title
            seen = set()
            unique_papers = []
            for p in papers:
                key = p.get('paper_number') or p['title']
                if key not in seen:
                    seen.add(key)
                    unique_papers.append(p)
            
            # Filter to only new papers (unless force)
            if not force and existing_papers:
                new_papers = [p for p in unique_papers if p.get('paper_number') not in existing_papers]
                self.logger.info(f"Found {len(new_papers)} new papers out of {len(unique_papers)}")
            else:
                new_papers = unique_papers
            
            if not new_papers and existing_papers:
                self.logger.info("No new NBER working papers to download")
                return self.data_dir / 'working_papers.json'
            
            # Merge with existing data
            existing_data = self.load_existing_json('working_papers.json')
            merged_data = self.merge_items(
                existing_data, new_papers, 'papers', 'paper_number'
            )
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['paper_count'] = len(merged_data.get('papers', []))
            
            return self.save_json(
                data=merged_data,
                filename='working_papers.json',
                source_url=url,
                metadata={'paper_count': merged_data['paper_count'], 'type': 'qualitative'},
                force=True  # Always save since we merged
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download working papers: {e}")
            return None
    
    def _download_digest(self, force: bool = False) -> Optional[Path]:
        """Download NBER Digest summaries incrementally."""
        try:
            # Get existing digest URLs to avoid re-downloading
            existing_urls = self.get_existing_keys('digest.json', 'digests', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing digests, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['digest']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            digests = []
            
            # Find digest articles
            for article in soup.find_all(['article', 'div'], class_=re.compile(r'(digest|article|post)')):
                title_elem = article.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = article.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    # Get summary
                    summary_elem = article.find(['p', 'div'], class_=re.compile(r'(summary|excerpt|description)'))
                    summary = summary_elem.get_text(strip=True) if summary_elem else None
                    
                    if title and len(title) > 10:
                        digests.append({
                            'title': title,
                            'url': link,
                            'summary': summary[:300] if summary else None
                        })
            
            # Deduplicate
            seen = set()
            unique_digests = []
            for d in digests:
                if d['title'] not in seen:
                    seen.add(d['title'])
                    unique_digests.append(d)
            
            # Filter to only new digests (unless force)
            if not force and existing_urls:
                new_digests = [d for d in unique_digests if d.get('url') not in existing_urls]
                self.logger.info(f"Found {len(new_digests)} new digests out of {len(unique_digests)}")
            else:
                new_digests = unique_digests
            
            if not new_digests and existing_urls:
                self.logger.info("No new NBER digests to download")
                return self.data_dir / 'digest.json'
            
            # Merge with existing data
            existing_data = self.load_existing_json('digest.json')
            merged_data = self.merge_items(
                existing_data, new_digests, 'digests', 'url'
            )
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['digest_count'] = len(merged_data.get('digests', []))
            
            return self.save_json(
                data=merged_data,
                filename='digest.json',
                source_url=url,
                metadata={'digest_count': merged_data['digest_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download digest: {e}")
            return None

