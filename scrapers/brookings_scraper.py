"""
Brookings Institution - Hutchins Center Scraper
================================================
Downloads fiscal policy research from the Hutchins Center including:
- Fiscal Impact Measure (FIM)
- Fiscal policy analysis and commentary

Hutchins Center: https://www.brookings.edu/centers/the-hutchins-center-on-fiscal-and-monetary-policy/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class BrookingsScraper(BaseScraper):
    """
    Scraper for Brookings Institution Hutchins Center data.
    
    Key products:
    - Fiscal Impact Measure (FIM): Measures fiscal policy contribution to GDP
    - Research papers on fiscal and monetary policy
    """
    
    SOURCE_NAME = 'brookings'
    BASE_URL = 'https://www.brookings.edu'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'hutchins': '/centers/the-hutchins-center-on-fiscal-and-monetary-policy/',
        'fim': '/interactives/hutchins-center-fiscal-impact-measure/',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available Brookings datasets."""
        return [
            {
                'name': 'Fiscal Impact Measure (FIM)',
                'key': 'fim',
                'description': 'Measures how much fiscal policy adds to or subtracts from GDP growth',
                'frequency': 'quarterly',
                'type': 'quantitative'
            },
            {
                'name': 'Hutchins Center Research',
                'key': 'hutchins_research',
                'description': 'Recent fiscal and monetary policy research papers',
                'frequency': 'ongoing',
                'type': 'qualitative'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific Brookings dataset."""
        if dataset_key == 'fim':
            return self._download_fim(force)
        elif dataset_key == 'hutchins_research':
            return self._download_hutchins_research(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_fim(self, force: bool = False) -> Optional[Path]:
        """
        Download Fiscal Impact Measure data.
        
        The FIM measures the contribution of fiscal policy to real GDP growth.
        Positive = fiscal stimulus, Negative = fiscal drag
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['fim']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the page content
            content = soup.get_text(separator='\n', strip=True)
            
            # Try to find data tables or charts
            tables = []
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                if rows:
                    tables.append(rows)
            
            # Look for any embedded data or charts
            scripts = soup.find_all('script')
            data_scripts = []
            for script in scripts:
                if script.string and ('data' in script.string.lower() or 'chart' in script.string.lower()):
                    data_scripts.append(script.string[:2000])  # Limit size
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'title': 'Fiscal Impact Measure (FIM)',
                'description': 'Measures fiscal policy contribution to GDP growth',
                'content_summary': content[:5000],
                'tables': tables,
                'word_count': len(content.split())
            }
            
            return self.save_json(
                data=data,
                filename='fim.json',
                source_url=url,
                metadata={'type': 'fiscal_measure'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download FIM: {e}")
            return None
    
    def _download_hutchins_research(self, force: bool = False) -> Optional[Path]:
        """Download recent Hutchins Center research papers incrementally."""
        try:
            # Get existing article URLs to avoid re-downloading
            existing_urls = self.get_existing_keys('hutchins_research.json', 'articles', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing articles, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['hutchins']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find research articles/papers
            articles = []
            for article in soup.find_all(['article', 'div'], class_=re.compile(r'(post|article|research)')):
                title_elem = article.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href') if title_elem.name == 'a' else None
                    if not link:
                        link_elem = article.find('a', href=True)
                        link = link_elem['href'] if link_elem else None
                    
                    # Get date if available
                    date_elem = article.find(class_=re.compile(r'date'))
                    date = date_elem.get_text(strip=True) if date_elem else None
                    
                    # Get summary/excerpt
                    summary_elem = article.find(['p', 'div'], class_=re.compile(r'(excerpt|summary|description)'))
                    summary = summary_elem.get_text(strip=True) if summary_elem else None
                    
                    article_url = link if link and link.startswith('http') else f"{self.BASE_URL}{link}" if link else None
                    
                    if title and len(title) > 10:
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'date': date,
                            'summary': summary
                        })
            
            # Deduplicate
            seen = set()
            unique_articles = []
            for a in articles:
                if a['title'] not in seen:
                    seen.add(a['title'])
                    unique_articles.append(a)
            
            # Filter to only new articles (unless force)
            if not force and existing_urls:
                new_articles = [a for a in unique_articles if a.get('url') not in existing_urls]
                self.logger.info(f"Found {len(new_articles)} new articles out of {len(unique_articles)}")
            else:
                new_articles = unique_articles
            
            if not new_articles and existing_urls:
                self.logger.info("No new Hutchins research to download")
                return self.data_dir / 'hutchins_research.json'
            
            # Merge with existing data
            existing_data = self.load_existing_json('hutchins_research.json')
            merged_data = self.merge_items(
                existing_data, new_articles, 'articles', 'url'
            )
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['article_count'] = len(merged_data.get('articles', []))
            
            return self.save_json(
                data=merged_data,
                filename='hutchins_research.json',
                source_url=url,
                metadata={'article_count': merged_data['article_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download Hutchins research: {e}")
            return None

