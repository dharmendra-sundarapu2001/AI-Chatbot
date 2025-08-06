import os
import requests
import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import unquote, parse_qs
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import langdetect
import random

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchService:
    def __init__(self):
        """Initialize the WebSearchService with proper configurations"""
        self.session = self._create_session()
        
        # We will now exclusively use a custom DuckDuckGo scraper
        self.search_providers = {
            'duckduckgo_scrape': self._search_with_duckduckgo_scraper,
        }
        
        self.duckduckgo_url = 'https://duckduckgo.com/html/?q='
        
        # A pool of user agents to rotate for better anonymity
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def is_web_search_query(self, query: str) -> bool:
        """
        (This method remains the same)
        """
        logger.info(f"🔍 WEB SEARCH DETECTION | Query: '{query}'")
        
        web_search_indicators = [
            'latest', 'recent', 'current', 'today', 'now', 'this week', 'this month', 'this year',
            'updated', 'new', 'breaking', 'fresh', 'this season', 'this time', '2024', '2025',
            'news', 'headlines', 'breaking news', 'current events', 'happening',
            'price', 'stock', 'weather', 'temperature', 'forecast',
            'exchange rate', 'currency', 'bitcoin', 'crypto',
            'search for', 'find information about', 'look up', 'google',
            'what is happening', 'what\'s new', 'tell me about recent',
            'github', 'stackoverflow', 'reddit', 'twitter', 'facebook',
            'wikipedia', 'youtube', 'instagram',
            'trending', 'viral', 'popular', 'top rated', 'best of',
            'review', 'comparison', 'versus', 'vs',
            'winner', 'champion', 'championship', 'tournament', 'final', 'match result',
            'ipl', 'world cup', 'olympics', 'premier league', 'nba', 'nfl', 'fifa',
            'score', 'result', 'standings', 'league table', 'season',
            'who is', 'what is', 'where is', 'when did', 'how to',
            'status of', 'information about', 'details about', 'who won', 'who wins'
        ]
        
        exclude_indicators = [
            'uploaded file', 'this file', 'document', 'pdf', 'above file',
            'according to', 'based on the file', 'from the document',
            'dvdrental', 'database', 'table', 'sql', 'query database'
        ]
        
        query_lower = query.lower()
        
        for exclude_term in exclude_indicators:
            if exclude_term in query_lower:
                logger.info(f"❌ WEB SEARCH EXCLUDED - Found exclusion term: '{exclude_term}'")
                return False
        
        for indicator in web_search_indicators:
            if indicator in query_lower:
                logger.info(f"✅ WEB SEARCH TRIGGERED - Matched indicator: '{indicator}'")
                return True
        
        question_patterns = [
            r'^what.*(latest|current|new|recent)',
            r'^who is.*(?!in|from|mentioned|discussed)',
            r'^who.*(winner|champion|won|wins)',
            r'^when did.*(?!the|this|mentioned)',
            r'^how.*(current|latest|now|today)',
            r'^where.*(current|latest|now|today)',
            r'winner.*season',
            r'champion.*\d{4}',
            r'ipl.*season',
            r'who won.*tournament',
            r'result.*match'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"✅ WEB SEARCH TRIGGERED - Matched pattern: '{pattern}'")
                return True
        
        logger.info(f"❌ WEB SEARCH NOT TRIGGERED - No matches found")
        return False
    
    def _get_headers(self):
        """Returns a random user agent from the list."""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://duckduckgo.com/' 
        }
        return headers

    def _search_with_duckduckgo_scraper(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Perform search by scraping DuckDuckGo's HTML page."""
        try:
            search_url = self.duckduckgo_url + requests.utils.quote(query)
            
            response = self.session.get(
                search_url,
                headers=self._get_headers(),
                timeout=10,
                verify=True
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Updated selectors based on current DuckDuckGo HTML structure
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:max_results]:
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if url.startswith('/l/?uddg='):
                        try:
                            parsed = parse_qs(url.split('?')[1])
                            if 'uddg' in parsed:
                                url = unquote(parsed['uddg'][0])
                        except Exception as e:
                            logger.warning(f"Failed to clean up DDG URL: {url} | Error: {e}")
                            
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
            
            return results
        except Exception as e:
            logger.error(f"❌ DuckDuckGo Scraper failed for query '{query}': {e}")
            return []
            
    def search_web(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Perform web search using only the DuckDuckGo scraper.
        """
        engine = 'duckduckgo_scrape'
        logger.info(f"🌐 WEB SEARCH INITIATED | Query: '{query}' | Using Engine: {engine}")
        start_time = time.time()
        
        try:
            results = self._search_with_duckduckgo_scraper(query, max_results)
            
            if not results:
                logger.warning(f"❌ SEARCH PROVIDER FAILED or ZERO RESULTS | Engine: {engine}.")
                # If no results from DDG, there is no fallback, so we fail
                return {
                    "status": "error",
                    "message": f"DuckDuckGo failed to return results for the query.",
                    "query": query,
                    "results": []
                }
            
            enriched_results = []
            for i, result in enumerate(results[:max_results]):
                if any(domain in result['url'] for domain in ['.cn', '.jp', '.ru', 'baidu.com', 'zhihu.com']):
                    logger.warning(f"⚠️ SKIPPING | Irrelevant domain detected: {result['url']}")
                    continue
                
                try:
                    time.sleep(random.uniform(1, 3))
                    content = self._scrape_webpage_content(result['url'])
                    
                    if content and ('en' in langdetect.detect(content[:500]) or not any(char.isalpha() for char in content[:500])):
                        result['scraped_content'] = content
                        enriched_results.append(result)
                        logger.info(f"📄 SCRAPED CONTENT | URL: {result['url'][:60]}... | Content length: {len(content)} chars")
                    else:
                        logger.warning(f"⚠️ SKIPPING | Non-English or empty content detected from {result['url']}")
                        result['scraped_content'] = result.get('snippet', '')
                        enriched_results.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ SCRAPING FAILED | URL: {result['url']} | Error: {str(e)}")
                    result['scraped_content'] = result.get('snippet', '')
                    enriched_results.append(result)

            if not enriched_results:
                 logger.warning(f"❌ ALL SCRAPED RESULTS FAILED | Engine: {engine}.")
                 return {
                    "status": "error",
                    "message": "Scraping failed for all results found by DuckDuckGo.",
                    "query": query,
                    "results": []
                }
            
            end_time = time.time()
            logger.info(f"✅ WEB SEARCH COMPLETED | Results: {len(enriched_results)} | Time: {end_time - start_time:.2f}s | Engine: {engine}")
            
            return {
                "status": "success",
                "query": query,
                "results": enriched_results,
                "search_time": end_time - start_time,
                "engine_used": engine
            }
        
        except Exception as e:
            logger.error(f"❌ WEB SEARCH UNEXPECTED ERROR | Engine: {engine} | Error: {str(e)}.")
            return {
                "status": "error",
                "message": f"Unexpected error during web search: {str(e)}",
                "query": query,
                "results": []
            }

    def _scrape_webpage_content(self, url: str, max_chars: int = 2000) -> str:
        """
        Scrape main content from a webpage.
        """
        try:
            scrape_headers = self._get_headers()
            scrape_headers['Referer'] = url
            
            response = self.session.get(
                url,
                headers=scrape_headers,
                timeout=8,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            content_selectors = [
                'article', 'main', '[role="main"]', '.content', '.main-content',
                '.post-content', '.entry-content', '.article-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True)
            
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            return content
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning(f"⚠️ Scraping failed for {url}: 403 Forbidden. Website actively blocked access.")
                return "SCRAPING FAILED: Website actively blocked access (403 Forbidden)."
            else:
                logger.warning(f"⚠️ Failed to scrape content from {url}: HTTP Error {e.response.status_code}")
                return f"SCRAPING FAILED: HTTP Error {e.response.status_code}."
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Failed to scrape content from {url}: {str(e)}")
            return f"SCRAPING FAILED: {str(e)}"

    def format_search_results_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results into a readable format for LLM processing.
        """
        if search_results["status"] != "success":
            return f"Web search failed: {search_results.get('message', 'Unknown error')}"
        
        results = search_results["results"]
        if not results:
            return "No relevant web search results found."
        
        formatted_results = f"Web Search Results for: '{search_results['query']}'\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"Result {i}:\n"
            formatted_results += f"Title: {result['title']}\n"
            formatted_results += f"URL: {result['url']}\n"
            
            if result.get('scraped_content') and not result['scraped_content'].startswith("SCRAPING FAILED"):
                content = result['scraped_content']
                if len(content) > 800:
                    content = content[:800] + "..."
                formatted_results += f"Content: {content}\n"
            elif result.get('snippet'):
                formatted_results += f"Snippet: {result['snippet']}\n"
            else:
                 formatted_results += f"Content: {result.get('scraped_content', 'Could not retrieve content.')}\n"
            
            formatted_results += "\n" + "-" * 50 + "\n\n"
        
        return formatted_results

    def test_web_search(self, test_query: str = "latest AI news") -> Dict[str, Any]:
        """
        Test the web search functionality
        """
        logger.info(f"🧪 TESTING WEB SEARCH | Query: '{test_query}'")
        return self.search_web(test_query, max_results=2)

# Create a singleton instance
web_search_service = WebSearchService()