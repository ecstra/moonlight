import asyncio
import logging
import warnings
from dataclasses import dataclass

import scrapy
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from scrapy.crawler import AsyncCrawlerRunner
from scrapy.signals import item_scraped

# Scrapy's reactor-less mode emits an "experimental httpx download handler" notice on
# every crawl, via Scrapy's logger (and as a warning). Silence both.
warnings.filterwarnings("ignore", message="HttpxDownloadHandler is experimental")
logging.getLogger("scrapy").setLevel(logging.ERROR)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    text: str = ""   # full page text, populated when fetch=True

# Per-result page text is capped at this many characters so a few very long pages
# don't blow up the token budget. Generous (well above a snippet); tune as needed.
MAX_RESULT_CHARS = 6000

# Speed-tuned settings for one-shot fetches. TWISTED_REACTOR_ENABLED=False runs
# the crawl under the current asyncio loop (AsyncCrawlerRunner + asyncio.run).
# Note: ROBOTSTXT_OBEY=False and the browser UA fetch more pages but are less
# polite; RETRY_ENABLED=False drops transiently-failing pages instead of retrying.
_SCRAPE_SETTINGS = {
    "TWISTED_REACTOR_ENABLED": False,
    "LOG_ENABLED": False,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "ROBOTSTXT_OBEY": False,
    "CONCURRENT_REQUESTS": 32,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 32,
    "DOWNLOAD_DELAY": 0.0,
    "DOWNLOAD_TIMEOUT": 15,
    "COOKIES_ENABLED": False,
    "RETRY_ENABLED": False,
}

class _PageSpider(scrapy.Spider):
    name = "moonlight_page_spider"

    def __init__(self, urls: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = urls or []

    def parse(self, response):
        # Pull text broadly: headings, paragraphs, lists, table cells, definition
        # descriptions, links, and spans. Version tables, spec lists, and profile bios
        # often live outside <p>/<li>, so a narrow selector silently drops the real
        # answer. MAX_RESULT_CHARS bounds the result, so extra nav/footer noise is fine.
        chunks = response.css(
            "h1::text, h2::text, h3::text, h4::text, p::text, li::text, "
            "td::text, th::text, dd::text, a::text, span::text"
        ).getall()
        yield {"url": response.url, "text": " ".join(" ".join(chunks).split())}

async def _ddg_search(
    query: str, 
    max_results: int
) -> list[dict]:
    """
    DuckDuckGo text search -> [{title, href, body}, ...].

    Returns an empty list when nothing is found. ddgs raises DDGSException instead
    of returning [] on no results (and on transient backend errors), but a search
    that finds nothing is a normal outcome, not a failure that should crash a run.
    """
    try:
        # ddgs is synchronous, so run it off the event loop.
        return await asyncio.to_thread(DDGS().text, query, max_results=max_results)
    except DDGSException:
        return []

async def _scrape(urls: list[str]) -> dict[str, str]:
    """
    Fetch each URL concurrently and return {url: extracted_text}.
    """
    texts: dict[str, str] = {}

    runner = AsyncCrawlerRunner(settings=_SCRAPE_SETTINGS)

    # weak=False: Scrapy holds signal receivers weakly, so a throwaway handler
    # would be garbage-collected before the crawl runs.
    def collect(item, **_):
        texts[item["url"]] = item["text"]

    crawler = runner.create_crawler(_PageSpider)
    crawler.signals.connect(collect, signal=item_scraped, weak=False)
    await runner.crawl(crawler, urls=urls)
    return texts

async def web_search(
    query: str,
    max_results: int = 5,
    fetch: bool = True
) -> str:
    """
    Search the web and return the results as formatted text for grounding.

    Uses DuckDuckGo (ddgs) for search and Scrapy to pull page text. Each result is
    rendered as "[n] title / url / text" with full (untruncated) page text. When
    fetch is False, the snippet is used instead of the page body.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
        fetch (bool): Whether to download and extract each page's text.

    Returns:
        str: The results rendered as one text block (empty string if none).
    """
    hits = await _ddg_search(query, max_results)
    results = [
        SearchResult(
            title=hit.get("title", ""),
            url=hit.get("href", ""),
            snippet=hit.get("body", ""),
        )
        for hit in hits if hit.get("href")
    ]

    if fetch and results:
        texts = await _scrape([r.url for r in results])
        for result in results:
            result.text = texts.get(result.url, "")

    if not results:
        return ""

    blocks = [f"Search results for {query!r}:", ""]
    for i, result in enumerate(results, 1):
        # Cap per-result text so a few huge pages don't blow the token budget.
        body = (result.text or result.snippet or "")[:MAX_RESULT_CHARS]
        blocks.append(f"[{i}] {result.title}\n{result.url}\n{body}\n")
    return "\n".join(blocks)