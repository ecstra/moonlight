import random, html2text, logging, asyncio, urllib
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

logger = logging.getLogger("hive")

class AsyncContentExtractor:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def extract_content(self, url):
        """Extract clean markdown content from a single URL"""
        async with self.semaphore:
            try:
                md_generator = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(threshold=0.5),
                    options={
                        "ignore_links": True,
                        "ignore_images": True,
                        "body_width": 0,
                        "escape_html": True
                    }
                )

                config = CrawlerRunConfig(
                    verbose=False,
                    cache_mode=CacheMode.BYPASS,
                    log_console=False,
                    markdown_generator=md_generator
                )

                browser_config = BrowserConfig(
                    verbose=False
                )
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url=url, config=config, verbose=False)
                    
                    if not result.success:
                        return {'url': url, 'content': None, 'error': result.error_message}
                    
                    # Use the filtered markdown content
                    content = getattr(result, 'markdown', '') or ''
                    if not content:
                        return {'url': url, 'content': None, 'error': 'No content found'}
                    
                    # Final length check
                    if len(content) < 500:
                        return {'url': url, 'content': None, 'error': 'Insufficient content'}
                        
                    return {
                        'url': url,
                        'content': content,
                        'error': None
                    }
                    
            except Exception as e:
                return {'url': url, 'content': None, 'error': str(e)}

    async def process_batch(self, urls):
        """Process URLs in parallel with optimized resource management"""
        tasks = [self.extract_content(url) for url in urls]
        results = []
        
        for i in range(0, len(tasks), 10):
            chunk = tasks[i:i + 10]
            try:
                chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
                valid_results = [r for r in chunk_results if isinstance(r, dict)]
                results.extend(valid_results)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                
        return results

class GoogleSearcher:
    def __init__(self, banned_links=None):
        self._setup_html2text()
        self.banned_links = set(banned_links or [])
        self.loop = asyncio.new_event_loop()
        
    def _setup_html2text(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.ignore_tables = True
        self.h2t.body_width = 0

    async def _fetch_search_results(self, query: str) -> str:
        query_encoded = quote_plus(query)
        search_url = f"https://www.google.com/search?q={query_encoded}"
        
        config = CrawlerRunConfig(
            verbose=False,
            cache_mode=CacheMode.BYPASS,
            log_console=False,
        )
        
        browser_config = BrowserConfig(
            verbose=False
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            search_result = await crawler.arun(url=search_url, config=config, verbose=False)
            return getattr(search_result, "html", "") or getattr(search_result, "markdown", "")


    def _extract_results(self, html_content: str) -> list:
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        seen_urls = set()

        # Try multiple Google result layouts
        search_divs = (
            soup.find_all("div", class_="g") +  # Standard results
            soup.find_all("div", class_="yuRUbf") +  # Alternative layout
            soup.find_all("div", attrs={"data-sokoban-container": True})  # Modern layout
        )

        for div in search_divs:
            try:
                # Find URL - check multiple possible locations
                url = None
                link = div.find('a', href=True)
                if link and 'href' in link.attrs:
                    url = link['href']
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    url = urllib.parse.unquote(url)
                
                if not url or 'google.com' in url or url in seen_urls or url in self.banned_links:
                    continue

                # Find title - check multiple possible locations
                title_element = (
                    div.find("h3") or 
                    div.find(attrs={"role": "heading"}) or
                    div.find(class_="LC20lb")
                )
                title = title_element.get_text().strip() if title_element else None
                
                if not title:
                    title = urlparse(url).netloc.split('.')[0].capitalize()

                # Find snippet if available
                snippet = None
                snippet_div = (
                    div.find(class_="VwiC3b") or  # Modern layout
                    div.find(class_="st") or      # Classic layout
                    div.find(class_="IsZvec")     # Alternative layout
                )
                if snippet_div:
                    snippet = snippet_div.get_text().strip()

                seen_urls.add(url)
                results.append({
                    'url': url, 
                    'title': title,
                    'snippet': snippet
                })

            except Exception as e:
                logger.warning(f"Error extracting result: {str(e)}")
                continue

        return results

    def search(self, query: str, num_results: int = 10):
        """
        Synchronous search method that handles async operations internally
        """
        
        try:
            # Create new event loop for this thread if necessary
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async search in the event loop
            results = loop.run_until_complete(self._async_search(query, num_results))
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    async def extract_contents_from_urls(self, urls: list[str]) -> list:
        """
        Extract content from a list of URLs asynchronously
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            List of dictionaries with 'url', 'content', and 'title' keys
        """
        try:
            # Process URLs in parallel using AsyncContentExtractor
            extractor = AsyncContentExtractor()
            content_results = await extractor.process_batch(urls)
            
            # Format successful extractions
            final_results = []
            for result in content_results:
                if result.get('content'):
                    final_results.append({
                        'url': result['url'],
                        'content': result['content'],
                        'title': urlparse(result['url']).netloc.split('.')[0].capitalize()
                    })
            
            logger.info(f"Successfully extracted content from {len(final_results)} of {len(urls)} URLs")
            return final_results
            
        except Exception as e:
            logger.error(f"Content extraction error: {str(e)}")
            return []
    
    def extract_contents(self, urls: list[str]) -> list:
        """
        Synchronous wrapper for extract_contents_from_urls
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            List of dictionaries with 'url', 'content', and 'title' keys
        """
        try:
            # Create or get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.extract_contents_from_urls(urls))
            
        except Exception as e:
            logger.error(f"Content extraction error: {str(e)}")
            return []
        
    async def _async_search(self, query: str, num_results: int = 10):
        """Internal async search method"""
        try:
            logger.info(f"Searching for: {query}")
            
            # Fetch search results
            html_content = await self._fetch_search_results(query)
            results = self._extract_results(html_content)
            
            if not results:
                logger.warning("No search results found")
                return []
            
            # Extract content using existing AsyncContentExtractor
            extractor = AsyncContentExtractor()
            content_results = await extractor.process_batch([r['url'] for r in results[:num_results]])
            
            # Merge results with fallback to snippets
            final_results = []
            for result in results[:num_results]:
                content_match = next((r for r in content_results if r['url'] == result['url']), None)
                
                if content_match and content_match['content']:
                    final_results.append({
                        'title': result['title'],
                        'url': result['url'],
                        'content': content_match['content']
                    })
                elif result.get('snippet'):
                    final_results.append({
                        'title': result['title'],
                        'url': result['url'],
                        'content': result['snippet']
                    })
            
            logger.info(f"Found {len(final_results)} valid results")
            return final_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

class WebSearch:
    def __init__(
            self, 
            query: str,
            urls: list[str] = [],
        ):
        self.query = query
        
        self.urls = urls
        
        if self.urls and self.query:
            raise ValueError("Please provide either a query or URLs, not both.")
        
        if not self.urls and not self.query:
            raise ValueError("Please provide either a query or URLs.")
        
        self.scraper = GoogleSearcher()
        
        self.results = []
        self.output = []
        
        self.search()
    
    def search(self) -> None:
        
        if self.urls:
            self.results = self.scraper.extract_contents(self.urls)
            
        elif self.query:
            self.results = self.scraper.search(
                self.query, 
                num_results=3
            )