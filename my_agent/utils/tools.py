# from langchain_community.tools.tavily_search import TavilySearchResults

# tools = [TavilySearchResults(max_results=1)]

from langchain_community.tools import DuckDuckGoSearchRun
tools = [DuckDuckGoSearchRun(max_results=3)]