import os
import requests
import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging configuration
logger = logging.getLogger(__name__)

class WebSearchService:
    def __init__(self):
        """Initialize the WebSearchService with proper configurations"""
        self.session = self._create_session()
        self.search_engines = {
            'duckduckgo': 'https://duckduckgo.com/html/?q=',
            'bing': 'https://www.bing.com/search?q='
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
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
        Determine if a query requires web search based on keywords and patterns
        """
        logger.info(f"🔍 WEB SEARCH DETECTION | Query: '{query}'")
        
        web_search_indicators = [
            # Time-sensitive queries
            'latest', 'recent', 'current', 'today', 'now', 'this week', 'this month', 'this year',
            'updated', 'new', 'breaking', 'fresh', 'this season', 'this time', '2024', '2025',
            
            # News and events
            'news', 'headlines', 'breaking news', 'current events', 'happening',
            
            # Real-time information
            'price', 'stock', 'weather', 'temperature', 'forecast',
            'exchange rate', 'currency', 'bitcoin', 'crypto',
            
            # Search intent words
            'search for', 'find information about', 'look up', 'google',
            'what is happening', 'what\'s new', 'tell me about recent',
            
            # Specific domains that change frequently
            'github', 'stackoverflow', 'reddit', 'twitter', 'facebook',
            'wikipedia', 'youtube', 'instagram',
            
            # Technology and trends
            'trending', 'viral', 'popular', 'top rated', 'best of',
            'review', 'comparison', 'versus', 'vs',
            
            # Sports and competitions (time-sensitive)
            'winner', 'champion', 'championship', 'tournament', 'final', 'match result',
            'ipl', 'world cup', 'olympics', 'premier league', 'nba', 'nfl', 'fifa',
            'score', 'result', 'standings', 'league table', 'season',
            
            # Questions that likely need current data
            'who is', 'what is', 'where is', 'when did', 'how to',
            'status of', 'information about', 'details about', 'who won', 'who wins'
        ]
        
        # Exclude queries that are clearly about uploaded files or internal data
        exclude_indicators = [
            'uploaded file', 'this file', 'document', 'pdf', 'above file',
            'according to', 'based on the file', 'from the document',
            'dvdrental', 'database', 'table', 'sql', 'query database'
        ]
        
        query_lower = query.lower()
        logger.info(f"🔍 Query (lowercase): '{query_lower}'")
        
        # Check for exclusion first
        for exclude_term in exclude_indicators:
            if exclude_term in query_lower:
                logger.info(f"❌ WEB SEARCH EXCLUDED - Found exclusion term: '{exclude_term}'")
                return False
        
        # Check for web search indicators
        matched_indicators = []
        for indicator in web_search_indicators:
            if indicator in query_lower:
                matched_indicators.append(indicator)
        
        if matched_indicators:
            logger.info(f"✅ WEB SEARCH TRIGGERED - Matched indicators: {matched_indicators}")
            return True
        
        # Additional pattern matching
        # Questions that start with common question words and seem to ask for current info
        question_patterns = [
            r'^what.*(latest|current|new|recent)',
            r'^who is.*(?!in|from|mentioned|discussed)',  # "who is" but not about documents
            r'^who.*(winner|champion|won|wins)',  # Sports winners
            r'^when did.*(?!the|this|mentioned)',
            r'^how.*(current|latest|now|today)',
            r'^where.*(current|latest|now|today)',
            r'winner.*season',  # "winner of this season"
            r'champion.*\d{4}',  # "champion 2024/2025"
            r'ipl.*season',     # IPL season related
            r'who won.*tournament',
            r'result.*match'
        ]
        
        matched_patterns = []
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                matched_patterns.append(pattern)
        
        if matched_patterns:
            logger.info(f"✅ WEB SEARCH TRIGGERED - Matched patterns: {matched_patterns}")
            return True
        
        logger.info(f"❌ WEB SEARCH NOT TRIGGERED - No matches found")
        return False

    def search_web(self, query: str, max_results: int = 3, engine: str = 'duckduckgo') -> Dict[str, Any]:
        """
        Perform web search and return structured results
        """
        logger.info(f"🌐 WEB SEARCH INITIATED | Query: '{query}' | Engine: {engine}")
        start_time = time.time()
        
        try:
            if engine not in self.search_engines:
                engine = 'duckduckgo'  # fallback
            
            search_url = self.search_engines[engine] + query
            
            response = self.session.get(
                search_url,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            if engine == 'duckduckgo':
                results = self._parse_duckduckgo_results(response.text, max_results)
            elif engine == 'bing':
                results = self._parse_bing_results(response.text, max_results)
            else:
                results = []
            
            # Scrape content from top results
            enriched_results = []
            for i, result in enumerate(results[:max_results]):
                try:
                    content = self._scrape_webpage_content(result['url'])
                    result['scraped_content'] = content
                    enriched_results.append(result)
                    logger.info(f"📄 SCRAPED CONTENT | URL: {result['url'][:60]}... | Content length: {len(content)} chars")
                except Exception as e:
                    logger.warning(f"⚠️ SCRAPING FAILED | URL: {result['url']} | Error: {str(e)}")
                    result['scraped_content'] = result.get('snippet', '')
                    enriched_results.append(result)
            
            end_time = time.time()
            logger.info(f"✅ WEB SEARCH COMPLETED | Results: {len(enriched_results)} | Time: {end_time - start_time:.2f}s")
            
            return {
                "status": "success",
                "query": query,
                "results": enriched_results,
                "search_time": end_time - start_time,
                "engine_used": engine
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ WEB SEARCH NETWORK ERROR | Query: '{query}' | Error: {str(e)}")
            return {
                "status": "error",
                "message": f"Network error during web search: {str(e)}",
                "query": query
            }
        except Exception as e:
            logger.error(f"❌ WEB SEARCH ERROR | Query: '{query}' | Error: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during web search: {str(e)}",
                "query": query
            }

    def _parse_duckduckgo_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo search results"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # DuckDuckGo HTML structure
        result_divs = soup.find_all('div', class_='result__body')
        
        for div in result_divs[:max_results]:
            try:
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    # Clean up the URL if it's a DuckDuckGo redirect
                    if url.startswith('/l/?uddg='):
                        # Extract the actual URL from DuckDuckGo's redirect
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(url.split('?')[1])
                        if 'uddg' in parsed:
                            url = urllib.parse.unquote(parsed['uddg'][0])
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
            except Exception as e:
                logger.warning(f"⚠️ Error parsing DuckDuckGo result: {e}")
                continue
        
        return results

    def _parse_bing_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Parse Bing search results"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Bing HTML structure
        result_divs = soup.find_all('li', class_='b_algo')
        
        for div in result_divs[:max_results]:
            try:
                title_elem = div.find('h2')
                url_elem = title_elem.find('a') if title_elem else None
                snippet_elem = div.find('div', class_='b_caption')
                
                if url_elem:
                    title = title_elem.get_text(strip=True)
                    url = url_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
            except Exception as e:
                logger.warning(f"⚠️ Error parsing Bing result: {e}")
                continue
        
        return results

    def _scrape_webpage_content(self, url: str, max_chars: int = 2000) -> str:
        """
        Scrape main content from a webpage
        """
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=8,
                allow_redirects=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article',
                'main',
                '[role="main"]',
                '.content',
                '.main-content',
                '.post-content',
                '.entry-content',
                '.article-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            # Fallback to body content if no specific content area found
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with single space
            content = content.strip()
            
            # Limit content length
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            return content
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to scrape content from {url}: {str(e)}")
            return ""

    def format_search_results_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results into a readable format for LLM processing
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
            
            if result.get('scraped_content'):
                content = result['scraped_content']
                # Limit content per result to keep it manageable
                if len(content) > 800:
                    content = content[:800] + "..."
                formatted_results += f"Content: {content}\n"
            elif result.get('snippet'):
                formatted_results += f"Snippet: {result['snippet']}\n"
            
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
