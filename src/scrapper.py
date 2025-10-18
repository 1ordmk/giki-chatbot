"""
Enhanced GIKI Web Scraper
Scrapes real GIKI website data with improved error handling
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGIKIScraper:
    """Enhanced scraper with better error handling and more data extraction"""
    
    def __init__(self, base_url: str = "https://giki.edu.pk"):
        self.base_url = base_url
        self.visited_urls = set()
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Selenium with better options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✓ Chrome WebDriver initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Chrome: {e}")
            logger.info("Please install Chrome and ChromeDriver")
            raise
    
    def scrape_page(self, url: str, use_selenium: bool = True) -> Dict:
        """Scrape a single page with fallback methods"""
        logger.info(f"Scraping: {url}")
        
        # Try Selenium first for dynamic content
        if use_selenium:
            try:
                return self._scrape_with_selenium(url)
            except Exception as e:
                logger.warning(f"Selenium failed: {e}, trying requests...")
        
        # Fallback to requests for static content
        try:
            return self._scrape_with_requests(url)
        except Exception as e:
            logger.error(f"Both methods failed for {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str) -> Dict:
        """Scrape using Selenium for JavaScript-heavy pages"""
        self.driver.get(url)
        
        # Wait for page to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            logger.warning(f"Timeout waiting for {url}")
        
        # Scroll to load lazy content
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except:
            pass
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        return self._extract_content(soup, url, method="selenium")
    
    def _scrape_with_requests(self, url: str) -> Dict:
        """Scrape using requests for static pages"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return self._extract_content(soup, url, method="requests")
    
    def _extract_content(self, soup: BeautifulSoup, url: str, method: str) -> Dict:
        """Extract and clean content from BeautifulSoup object"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "Untitled"
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content_text = main_content.get_text(separator=' ', strip=True)
        else:
            content_text = soup.get_text(separator=' ', strip=True)
        
        # Clean text
        content_text = self._clean_text(content_text)
        
        # Extract metadata
        category = self._categorize_url(url)
        
        # Extract all links
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(url, href)
            if self._is_valid_giki_url(absolute_url):
                links.append(absolute_url)
        
        return {
            "url": url,
            "title": title_text,
            "category": category,
            "content": content_text,
            "links": list(set(links)),  # Remove duplicates
            "method": method,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"\/]', '', text)
        
        # Remove very short words (likely artifacts)
        words = text.split()
        words = [w for w in words if len(w) > 1 or w in ['.', ',', '!', '?']]
        text = ' '.join(words)
        
        return text.strip()
    
    def _categorize_url(self, url: str) -> str:
        """Categorize URL based on path"""
        url_lower = url.lower()
        
        if any(word in url_lower for word in ['admission', 'apply', 'entry']):
            return 'admissions'
        elif any(word in url_lower for word in ['academic', 'department', 'faculty', 'program']):
            return 'academics'
        elif any(word in url_lower for word in ['campus', 'student', 'society', 'club', 'hostel', 'sport']):
            return 'student_life'
        elif any(word in url_lower for word in ['research', 'publication', 'lab']):
            return 'research'
        elif any(word in url_lower for word in ['about', 'history', 'contact', 'vision']):
            return 'about'
        else:
            return 'general'
    
    def _is_valid_giki_url(self, url: str) -> bool:
        """Check if URL should be scraped"""
        parsed = urlparse(url)
        
        # Must be GIKI domain
        if 'giki.edu.pk' not in parsed.netloc:
            return False
        
        # Skip files
        skip_extensions = ['.pdf', '.jpg', '.png', '.jpeg', '.zip', '.doc', '.docx', '.xls', '.xlsx']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip social media and external links
        skip_patterns = ['facebook', 'twitter', 'youtube', 'linkedin', 'instagram', 'mailto:', 'tel:']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        return True
    
    def scrape_targeted_pages(self) -> List[Dict]:
        """Scrape important GIKI pages with known URLs"""
        
        target_pages = {
            # Admissions
            "Undergraduate Admissions": "https://giki.edu.pk/admissions/undergraduate/",
            "Graduate Admissions": "https://giki.edu.pk/admissions/graduate/",
            "Admission Requirements": "https://giki.edu.pk/admissions/requirements/",
            "Fee Structure": "https://giki.edu.pk/admissions/fee-structure/",
            
            # Academics
            "Faculties Overview": "https://giki.edu.pk/academics/",
            "Computer Science": "https://giki.edu.pk/faculty-of-computer-science/",
            "Electrical Engineering": "https://giki.edu.pk/faculty-of-electrical-engineering/",
            "Mechanical Engineering": "https://giki.edu.pk/faculty-of-mechanical-engineering/",
            
            # Student Life
            "Campus Life": "https://giki.edu.pk/campus-life/",
            "Student Societies": "https://giki.edu.pk/student-societies/",
            "Hostels": "https://giki.edu.pk/hostel-facilities/",
            "Sports": "https://giki.edu.pk/sports/",
            
            # About
            "About GIKI": "https://giki.edu.pk/about/",
            "History": "https://giki.edu.pk/about/history/",
            "Contact": "https://giki.edu.pk/contact/",
        }
        
        scraped_data = []
        total = len(target_pages)
        
        for i, (name, url) in enumerate(target_pages.items(), 1):
            logger.info(f"\n[{i}/{total}] Scraping: {name}")
            
            if url in self.visited_urls:
                logger.info("Already visited, skipping...")
                continue
            
            try:
                data = self.scrape_page(url, use_selenium=True)
                
                if data and len(data['content']) > 100:
                    scraped_data.append(data)
                    self.visited_urls.add(url)
                    logger.info(f"✓ Scraped {len(data['content'])} characters")
                else:
                    logger.warning(f"✗ Insufficient content from {name}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"✗ Failed to scrape {name}: {e}")
                continue
        
        return scraped_data
    
    def crawl_from_seed(self, seed_urls: List[str], max_pages: int = 50) -> List[Dict]:
        """Crawl starting from seed URLs"""
        to_visit = seed_urls.copy()
        scraped_data = []
        
        while to_visit and len(scraped_data) < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            logger.info(f"\n[{len(scraped_data)+1}/{max_pages}] Crawling: {url}")
            
            try:
                data = self.scrape_page(url, use_selenium=False)  # Use requests for speed
                
                if data and len(data['content']) > 200:
                    scraped_data.append(data)
                    self.visited_urls.add(url)
                    
                    # Add new links to queue
                    for link in data['links']:
                        if link not in self.visited_urls and link not in to_visit:
                            to_visit.append(link)
                    
                    logger.info(f"✓ Scraped successfully, found {len(data['links'])} links")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"✗ Error: {e}")
                self.visited_urls.add(url)  # Mark as visited to avoid retry
                continue
        
        return scraped_data
    
    def save_data(self, data: List[Dict], filename: str = "giki_scraped.json"):
        """Save scraped data to JSON"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ Saved {len(data)} pages to {filepath}")
        
        # Also save summary
        summary = {
            "total_pages": len(data),
            "categories": {},
            "total_content_length": sum(len(d['content']) for d in data),
            "urls": [d['url'] for d in data]
        }
        
        for doc in data:
            cat = doc['category']
            summary['categories'][cat] = summary['categories'].get(cat, 0) + 1
        
        summary_path = self.data_dir / filename.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Saved summary to {summary_path}")
        logger.info(f"\nCategory breakdown:")
        for cat, count in summary['categories'].items():
            logger.info(f"  - {cat}: {count} pages")
    
    def close(self):
        """Close browser"""
        try:
            self.driver.quit()
            logger.info("✓ Browser closed")
        except:
            pass


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🕷️  GIKI ENHANCED WEB SCRAPER")
    print("="*70 + "\n")
    
    scraper = EnhancedGIKIScraper()
    
    try:
        # Choose scraping method
        print("Choose scraping method:")
        print("1. Targeted pages (recommended, ~15 pages)")
        print("2. Broad crawl (slower, up to 50 pages)")
        print("3. Both (most comprehensive)")
        
        choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
        
        all_data = []
        
        if choice in ["1", "3"]:
            print("\n📍 Starting targeted scraping...")
            targeted_data = scraper.scrape_targeted_pages()
            all_data.extend(targeted_data)
            print(f"\n✓ Scraped {len(targeted_data)} targeted pages")
        
        if choice in ["2", "3"]:
            print("\n🌐 Starting broad crawl...")
            seed_urls = [
                "https://giki.edu.pk",
                "https://giki.edu.pk/admissions/",
                "https://giki.edu.pk/academics/"
            ]
            crawled_data = scraper.crawl_from_seed(seed_urls, max_pages=50)
            all_data.extend(crawled_data)
            print(f"\n✓ Crawled {len(crawled_data)} additional pages")
        
        # Remove duplicates based on URL
        unique_data = {d['url']: d for d in all_data}
        final_data = list(unique_data.values())
        
        # Save data
        scraper.save_data(final_data, "giki_scraped.json")
        
        print("\n" + "="*70)
        print(f"✅ SCRAPING COMPLETE!")
        print(f"📊 Total pages: {len(final_data)}")
        print(f"📁 Saved to: data/raw/giki_scraped.json")
        print("\n💡 Next steps:")
        print("   1. Run: python preprocessing.py")
        print("   2. Start backend: python backend.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    main()