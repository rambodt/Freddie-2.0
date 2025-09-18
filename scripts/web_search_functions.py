#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
web_search_functions.py
=======================

Web search and real-time utilities for Freddie’s function-calling layer.

Features
--------
- Web search via SerpAPI (Google) when `SERPAPI_KEY` is set; otherwise a free
  DuckDuckGo Instant Answer fallback with light post-processing.
- Weather lookup using free endpoints (wttr.in → Open-Meteo with Nominatim geocoding).
- Time/date helper and a safe arithmetic evaluator for simple expressions.
- Function tool schema (`FUNCTION_TOOLS`) for LLM integrations (e.g., Gemini).

Environment Variables
---------------------
- SERPAPI_KEY        : API key for SerpAPI (Google results).
- OPENWEATHER_API_KEY: Reserved; not required by current implementation.

PowerShell Example
------------------
 PS> $env:SERPAPI_KEY="YOUR_KEY"
 PS> python .\web_search_functions.py
"""

from __future__ import annotations
import os
import json
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests

__author__  = "Rambod Taherian"
__email__   = "rambodt@uw.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__     = "https://github.com/rambodt/Freddie-2.0"

JSONDict = Dict[str, Any]


class WebSearchFunctions:
    """Web search and real-time data retrieval helpers."""

    def __init__(self) -> None:
        self.serpapi_key: str = os.environ.get("SERPAPI_KEY", "")
        self.weather_api_key: str = os.environ.get("OPENWEATHER_API_KEY", "")

        print(f"[WebSearch Init] SerpAPI key: {'Configured' if self.serpapi_key else 'Not found'}")

    # ─────────────────────────── Search (public) ────────────────────────────

    def search_web(self, query: str, num_results: int = 3) -> JSONDict:
        """
        Search the web for `query`.

        Selection
        ---------
        - If `SERPAPI_KEY` is configured → use SerpAPI (Google).
        - Otherwise → use a free DuckDuckGo Instant Answer fallback.

        Parameters
        ----------
        query : str
            Search string.
        num_results : int, default 3
            Maximum number of results to return.

        Returns
        -------
        dict
            { "query": str, "results": list[dict], "timestamp": ISO8601, ... }
        """
        print(f"[search_web] Query: {query}")
        print(f"[search_web] SerpAPI available: {bool(self.serpapi_key)}")
        try:
            if self.serpapi_key:
                return self._search_serpapi(query, num_results)
            return self.search_web_free(query, num_results)
        except Exception as e:
            print(f"[search_web] Error: {e}")
            return self.search_web_free(query, num_results)

    def search_web_free(self, query: str, num_results: int = 3) -> JSONDict:
        """
        Free web search using DuckDuckGo Instant Answer API with light processing.

        Parameters
        ----------
        query : str
            Search string.
        num_results : int, default 3
            Maximum number of results to return.

        Returns
        -------
        dict
            A normalized result payload with up to `num_results` entries.
        """
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results: List[JSONDict] = []

                # Instant answer
                abstract = data.get("AbstractText") or ""
                if abstract:
                    results.append({
                        "title": "Summary",
                        "snippet": abstract[:500],
                        "link": data.get("AbstractURL", "")
                    })

                # Quick answer
                answer = data.get("Answer") or ""
                if answer:
                    results.append({
                        "title": "Quick Answer",
                        "snippet": answer,
                        "link": ""
                    })

                # Definition
                definition = data.get("Definition") or ""
                if definition:
                    results.append({
                        "title": "Definition",
                        "snippet": definition,
                        "link": data.get("DefinitionURL", "")
                    })

                # Related topics
                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        text = topic["Text"]
                        if len(text) > 20:
                            results.append({
                                "title": "Related Info",
                                "snippet": text[:300],
                                "link": topic.get("FirstURL", "")
                            })

                if results:
                    return {
                        "query": query,
                        "results": results[:num_results],
                        "timestamp": datetime.now().isoformat()
                    }

            # Simple template when the instant answer API yields nothing useful
            return {
                "query": query,
                "results": [{
                    "title": "Limited search results",
                    "snippet": f"I found limited information about '{query}'. You can also try a direct search.",
                    "link": ""
                }],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[search_web_free] Error: {e}")
            return {
                "query": query,
                "results": [{
                    "title": "Search error",
                    "snippet": f"Search encountered an error: {str(e)}",
                    "link": ""
                }],
                "timestamp": datetime.now().isoformat()
            }

    # ────────────────────────── Search (providers) ───────────────────────────

    def _search_serpapi(self, query: str, num_results: int) -> JSONDict:
        """
        SerpAPI-backed Google search.

        Parameters
        ----------
        query : str
            Search string.
        num_results : int
            Maximum count of organic results to include.

        Returns
        -------
        dict
            Normalized result payload; falls back to `search_web_free` if empty.
        """
        try:
            url = "https://serpapi.com/search"
            params = {"q": query, "api_key": self.serpapi_key, "num": num_results, "engine": "google"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results: List[JSONDict] = []

                # Answer box
                abox = data.get("answer_box") or {}
                if isinstance(abox, dict):
                    if abox.get("answer"):
                        results.append({"title": "Direct Answer", "snippet": abox["answer"], "link": ""})
                    elif abox.get("snippet"):
                        results.append({"title": "Quick Answer", "snippet": abox["snippet"], "link": abox.get("link", "")})

                # Knowledge graph
                kg = data.get("knowledge_graph") or {}
                if isinstance(kg, dict) and kg.get("description"):
                    results.append({"title": kg.get("title", "Info"), "snippet": kg["description"], "link": kg.get("website", "")})

                # Organic results
                for item in (data.get("organic_results") or [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    })

                if results:
                    print(f"[SerpAPI] Found {len(results)} results for: {query}")
                    return {
                        "query": query,
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "source": "SerpAPI/Google"
                    }

                print("[SerpAPI] No results, falling back")
                return self.search_web_free(query, num_results)

            print(f"[SerpAPI] HTTP {response.status_code}, falling back")
            return self.search_web_free(query, num_results)

        except Exception as e:
            print(f"[SerpAPI] Exception: {e}")
            return self.search_web_free(query, num_results)

    def _search_fallback(self, query: str) -> JSONDict:
        """
        Legacy DuckDuckGo Instant Answer fallback.

        Parameters
        ----------
        query : str
            Search string.

        Returns
        -------
        dict
            Normalized result payload.
        """
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results: List[JSONDict] = []

                abstract = data.get("AbstractText") or ""
                if abstract:
                    results.append({"title": "Summary", "snippet": abstract, "link": data.get("AbstractURL", "")})

                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "snippet": topic["Text"],
                            "link": topic.get("FirstURL", "")
                        })

                if not results:
                    results.append({
                        "title": "No direct results",
                        "snippet": f"Try searching for '{query}' on Google or other search engines",
                        "link": f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                    })

                return {"query": query, "results": results, "timestamp": datetime.now().isoformat()}

        except Exception:
            pass

        return {
            "query": query,
            "results": [{"title": "Search unavailable", "snippet": "Unable to perform web search at this time", "link": ""}],
            "timestamp": datetime.now().isoformat()
        }

    # ───────────────────────────── Weather ────────────────────────────────

    def get_weather_free(self, location: str = "Seattle") -> JSONDict:
        """
        Weather via free endpoints: wttr.in, then Open-Meteo with Nominatim geocoding.

        Parameters
        ----------
        location : str, default "Seattle"
            City or query text (e.g., "Los Angeles", "Paris").

        Returns
        -------
        dict
            Compact weather object with description, temperature, wind, and timestamp.
        """
        try:
            clean = location.strip().replace(" ", "+")

            # wttr.in compact text format
            simple_url = f"http://wttr.in/{clean}?format=%C+%t+%h+%w"
            resp = requests.get(simple_url, timeout=10, headers={'User-Agent': 'curl/7.68.0'})
            if resp.status_code == 200 and not resp.text.startswith("<!"):
                text = resp.text.strip()
                parts = text.split()
                temp = "Unknown"
                for part in parts:
                    if "°" in part or part.endswith("F") or part.endswith("C"):
                        temp = part
                        break
                return {
                    "location": clean.replace("+", " "),
                    "temperature": temp,
                    "feels_like": temp,
                    "description": text,
                    "humidity": "N/A",
                    "wind_speed": "N/A",
                    "timestamp": datetime.now().isoformat()
                }

            # Open-Meteo fallback with geocoding via Nominatim
            geo_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(clean)}&format=json&limit=1"
            g = requests.get(geo_url, headers={'User-Agent': 'Freddie-Robot/1.0'}, timeout=5)
            if g.status_code == 200:
                geo = g.json()
                if geo:
                    lat = geo[0]['lat']
                    lon = geo[0]['lon']
                    weather_url = (
                        "https://api.open-meteo.com/v1/forecast"
                        f"?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"
                    )
                    w = requests.get(weather_url, timeout=5)
                    if w.status_code == 200:
                        wd = w.json()
                        current = wd.get('current_weather', {})
                        code = int(current.get('weathercode', 0))
                        code_map = {
                            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                            45: "Foggy", 48: "Foggy", 51: "Light drizzle", 53: "Moderate drizzle",
                            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain",
                            65: "Heavy rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow"
                        }
                        desc = code_map.get(code, "Unknown conditions")
                        return {
                            "location": clean.replace("+", " "),
                            "temperature": f"{current.get('temperature', '?')}°F",
                            "feels_like": f"{current.get('temperature', '?')}°F",
                            "description": desc,
                            "humidity": "N/A",
                            "wind_speed": f"{current.get('windspeed', '?')} mph",
                            "timestamp": datetime.now().isoformat()
                        }

            return {
                "location": clean.replace("+", " "),
                "temperature": "Unable to fetch",
                "description": "Weather service temporarily unavailable",
                "error": "Could not connect to weather services"
            }

        except Exception as e:
            print(f"[Weather] Error: {e}")
            return {
                "location": location,
                "temperature": "Error",
                "description": f"Weather lookup failed: {str(e)}",
                "error": str(e)
            }

    # ───────────────────────────── Time/Date ─────────────────────────────

    def get_time_date(self, timezone: Optional[str] = None) -> JSONDict:
        """
        Current local date/time (server-local).

        Parameters
        ----------
        timezone : str | None
            Placeholder for future timezone support.

        Returns
        -------
        dict
            Date, time, ISO timestamp, and weekday name.
        """
        now = datetime.now()
        return {
            "date": now.strftime("%A, %B %d, %Y"),
            "time": now.strftime("%I:%M %p"),
            "timestamp": now.isoformat(),
            "day_of_week": now.strftime("%A")
        }

    # ───────────────────────────── Calculator ─────────────────────────────

    def calculate(self, expression: str) -> JSONDict:
        """
        Evaluate a simple arithmetic expression safely.

        Supported operators
        -------------------
        +, -, *, /, %, ** and unary minus.

        Parameters
        ----------
        expression : str
            Mathematical expression (e.g., "3*(2+5)**2 / 7").

        Returns
        -------
        dict
            Result object with original expression and numeric result, or error.
        """
        try:
            import ast
            import operator as op

            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Pow: op.pow,
                ast.Mod: op.mod,
                ast.USub: op.neg,
            }

            def eval_expr(node: ast.AST) -> float:
                if isinstance(node, ast.Num):  # < Py3.8
                    return node.n  # type: ignore[attr-defined]
                if isinstance(node, ast.Constant):  # Py3.8+
                    if isinstance(node.value, (int, float)):
                        return node.value  # type: ignore[return-value]
                    raise ValueError("Unsupported constant")
                if isinstance(node, ast.BinOp) and type(node.op) in operators:
                    return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                    return operators[type(node.op)](eval_expr(node.operand))
                raise ValueError("Unsupported expression")

            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)  # type: ignore[arg-type]
            return {"expression": expression, "result": result, "formatted": f"{expression} = {result}"}

        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}", "expression": expression}


# ───────────────────────── Function tool schema ──────────────────────────────

FUNCTION_TOOLS: List[Dict[str, Any]] = [
    {
        "function_declarations": [
            {
                "name": "search_web",
                "description": "Search the web for current information about any topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_weather_free",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "get_time_date",
                "description": "Get current time and date",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    }
]


if __name__ == "__main__":
    # Minimal demo for manual testing (PowerShell-friendly).
    ws = WebSearchFunctions()
    print(json.dumps(ws.search_web("University of Washington"), indent=2)[:800], "...\n")
    print(json.dumps(ws.get_weather_free("Seattle"), indent=2))
    print(json.dumps(ws.get_time_date(), indent=2))
    print(json.dumps(ws.calculate("3*(2+5)**2 / 7"), indent=2))
