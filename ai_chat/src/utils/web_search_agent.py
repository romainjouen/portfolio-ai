from __future__ import annotations
from typing import List, Dict, Optional, Any, Callable, Protocol
from pydantic import BaseModel, Field
from functools import wraps
from dataclasses import dataclass
from enum import Enum, auto
import requests
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
import os
import time
import datetime
import streamlit as st
from src.config.config import Config
from .llm_provider import LLMProvider

class SearchEngine(str, Enum):
    """Supported search engines with their API configurations"""
    BRAVE = 'brave'
    DUCKDUCKGO = 'duckduckgo'
    GOOGLE = 'google'
    SERPAPI = 'serpapi'
    SERPER = 'serper'

    @property
    def requires_api_key(self) -> bool:
        """Check if the search engine requires an API key"""
        return self != SearchEngine.DUCKDUCKGO

    @property
    def api_config(self) -> Dict[str, str]:
        """Get API configuration for the search engine"""
        configs = {
            SearchEngine.BRAVE: {
                'url': 'https://api.search.brave.com/res/v1/web/search',
                'key_name': 'brave_api_key',
                'header_key': 'X-Subscription-Token'
            },
            SearchEngine.SERPAPI: {
                'url': 'https://serpapi.com/search',
                'key_name': 'serpapi_api_key',
                'param_key': 'api_key'
            },
            SearchEngine.SERPER: {
                'url': 'https://google.serper.dev/search',
                'key_name': 'serper_api_key',
                'header_key': 'X-API-KEY'
            }
        }
        return configs.get(self, {})

@dataclass
class SearchCredentials:
    """Search engine credentials container"""
    brave_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    serper_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> SearchCredentials:
        """Create credentials from environment variables"""
        return cls(**{
            'brave_api_key': os.getenv("BRAVE_API_KEY"),
            'google_api_key': os.getenv("GOOGLE_API_KEY"),
            'google_cse_id': os.getenv("GOOGLE_CUSTOM_SEARCH_ID"),
            'serper_api_key': os.getenv("SERPER_API_KEY"),
            'serpapi_api_key': os.getenv("SERPAPI_API_KEY")
        })

    def get_key(self, key_name: str) -> Optional[str]:
        """Get API key by name"""
        return getattr(self, key_name, None)

class SearchResult(BaseModel):
    """Search result model with validation"""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Snippet/description of the search result")
    source: str = Field(..., description="Source search engine")

    class Config:
        frozen = True

def log_message(level: str, message: str) -> None:
    """Print a formatted log message with timestamp"""
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - {level} - {message}")

def rate_limit(seconds: int = 1):
    """Decorator for rate limiting API calls"""
    def decorator(func):
        last_called = {}
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            if func.__name__ in last_called:
                elapsed = current_time - last_called[func.__name__]
                if elapsed < seconds:
                    time.sleep(seconds - elapsed)
            last_called[func.__name__] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class WebSearchAgent:
    """Agent for performing web searches across multiple search engines"""
    
    def __init__(self, search_engines: List[str], results_limit: int = 5):
        self.results_limit = results_limit
        self.credentials = SearchCredentials.from_env()
        self.config = Config()
        self.llm = LLMProvider(selected_provider=st.session_state.current_provider)
        self.session = requests.Session()
        self.search_clients = self._initialize_search_clients(search_engines)

    def _initialize_search_clients(self, search_engines: List[str]) -> Dict[str, Callable]:
        """Initialize search clients based on available credentials"""
        clients = {}
        
        for engine in search_engines:
            engine_enum = SearchEngine(engine)
            if not engine_enum.requires_api_key:
                clients[engine] = self._duckduckgo_search
                continue

            if engine_enum == SearchEngine.GOOGLE:
                if self.credentials.google_api_key and self.credentials.google_cse_id:
                    try:
                        self.google_service = build(
                            "customsearch", "v1", 
                            developerKey=self.credentials.google_api_key
                        )
                        clients[engine] = self._google_search
                        log_message("INFO", f"{engine} API initialized successfully")
                    except Exception as e:
                        log_message("ERROR", f"Error initializing {engine} API: {e}")
                continue

            api_config = engine_enum.api_config
            if api_config and self.credentials.get_key(api_config['key_name']):
                clients[engine] = getattr(self, f"_{engine}_search")
                log_message("INFO", f"{engine} API initialized successfully")
            else:
                log_message("WARNING", f"{engine} credentials not found")

        return clients

    async def process(self, query: str) -> List[Dict[str, str]]:
        """Process search query across all configured search engines"""
        all_results = []
        
        for engine, search_func in self.search_clients.items():
            try:
                results = await search_func(query)
                all_results.extend([result.dict() for result in results])
            except Exception as e:
                log_message("ERROR", f"Error with {engine}: {str(e)}")
        
        return self._deduplicate_results(all_results)[:self.results_limit]

    def _make_api_request(self, engine: SearchEngine, query: str, method: str = 'get', **kwargs) -> List[SearchResult]:
        """Make API request to search engine"""
        api_config = engine.api_config
        if not api_config:
            return []

        try:
            request_kwargs = {
                'url': api_config['url'],
                **kwargs
            }

            if 'header_key' in api_config:
                request_kwargs.setdefault('headers', {})
                request_kwargs['headers'][api_config['header_key']] = self.credentials.get_key(api_config['key_name'])
            elif 'param_key' in api_config:
                request_kwargs.setdefault('params', {})
                request_kwargs['params'][api_config['param_key']] = self.credentials.get_key(api_config['key_name'])

            response = getattr(self.session, method)(**request_kwargs)
            response.raise_for_status()
            return self._parse_response(engine, response.json())
        except Exception as e:
            log_message("ERROR", f"{engine} search error: {str(e)}")
            return []

    def _parse_response(self, engine: SearchEngine, data: Dict) -> List[SearchResult]:
        """Parse API response into SearchResult objects"""
        parsers = {
            SearchEngine.BRAVE: lambda d: [
                SearchResult(
                    title=r["title"],
                    url=r["url"],
                    snippet=r["description"],
                    source=engine
                )
                for r in d.get("web", {}).get("results", [])
            ],
            SearchEngine.SERPAPI: lambda d: [
                SearchResult(
                    title=r.get('title', ''),
                    url=r.get('link', ''),
                    snippet=r.get('snippet', ''),
                    source=engine
                )
                for r in d.get('organic_results', [])
            ],
            SearchEngine.SERPER: lambda d: [
                SearchResult(
                    title=r.get('title', ''),
                    url=r.get('link', ''),
                    snippet=r.get('snippet', ''),
                    source=engine
                )
                for r in d.get('organic', [])
            ]
        }
        return parsers.get(engine, lambda _: [])(data)

    @rate_limit(seconds=1)
    async def _brave_search(self, query: str) -> List[SearchResult]:
        """Perform search using Brave Search API"""
        return self._make_api_request(
            SearchEngine.BRAVE,
            query,
            params={"q": query, "count": self.results_limit}
        )

    async def _duckduckgo_search(self, query: str) -> List[SearchResult]:
        """Perform search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                return [
                    SearchResult(
                        title=r["title"],
                        url=r["link"],
                        snippet=r["body"],
                        source=SearchEngine.DUCKDUCKGO
                    )
                    for r in ddgs.text(query, max_results=self.results_limit)
                    if isinstance(r, dict) and all(k in r for k in ('title', 'link', 'body'))
                ]
        except Exception as e:
            log_message("ERROR", f"DuckDuckGo search error: {str(e)}")
            return []

    @rate_limit(seconds=1)
    async def _google_search(self, query: str) -> List[SearchResult]:
        """Perform search using Google Custom Search API"""
        results = []
        pages_to_search = (self.results_limit + 9) // 10
        
        try:
            for i in range(pages_to_search):
                start_index = i * 10 + 1
                response = self.google_service.cse().list(
                    q=query,
                    cx=self.credentials.google_cse_id,
                    start=start_index,
                    num=min(10, self.results_limit - len(results))
                ).execute()

                if 'items' in response:
                    results.extend([
                        SearchResult(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source=SearchEngine.GOOGLE
                        )
                        for item in response['items']
                    ])
                
                if len(results) >= self.results_limit:
                    break

            return results[:self.results_limit]
        except Exception as e:
            log_message("ERROR", f"Google search error: {str(e)}")
            return []

    @rate_limit(seconds=1)
    async def _serpapi_search(self, query: str) -> List[SearchResult]:
        """Perform search using SerpAPI"""
        return self._make_api_request(
            SearchEngine.SERPAPI,
            query,
            params={
                'q': query,
                'num': self.results_limit,
                'engine': 'google'
            }
        )

    @rate_limit(seconds=1)
    async def _serper_search(self, query: str) -> List[SearchResult]:
        """Perform search using Serper"""
        return self._make_api_request(
            SearchEngine.SERPER,
            query,
            method='post',
            headers={'Content-Type': 'application/json'},
            json={"q": query, "num": self.results_limit}
        )

    def _deduplicate_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        return [
            result for result in results
            if result['url'] not in seen_urls and not seen_urls.add(result['url'])
        ] 