import requests
from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import settings

tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

@tool
def search_recipes(ingredients: list[str]) -> list[dict]:
    """
        Search Spoonacular for recipes using the given ingredients.
        Best for common/international dishes. Spoonacular's coverage of
        African/traditional cuisine is thin — if this returns few or no
        results for a local dish request, use search_recipes_web instead.

        Args:
            ingredients: list of ingredient names on hand, e.g. ["chicken", "rice"]

        Returns:
            List of recipe dicts with name, used/missing ingredient counts,
            and a source URL.
    """

    resp = requests.get(
        "https://api.spoonacular.com/recipes/findByIngredients",
        params={
            "apiKey": settings.SPOONACULAR_API_KEY,
            "ingredients": ",".join(ingredients),
            "number": 5,  # limit to top 5 results
            "ranking": 1,  # maximize used ingredients
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "id": r["id"],
            "name": r["title"],
            "used_ingredients": [i["name"] for i in r.get("usedIngredients", [])],
            "missing_ingredients": [i["name"] for i in r.get("missedIngredients", [])],
            "source": f"https://spoonacular.com/recipes/{r['title'].replace(' ', '-').lower()}-{r['id']}",
        }
        for r in data
    ]

@tool
def search_recipes_web(query: str) -> list[dict]:
    """
        Search the web for recipes when Spoonacular's results are thin —
    typically for traditional/African dishes it doesn't cover well.
    Use a natural search phrase, e.g. "sukuma wiki recipe with chicken".

    Args:
        query: natural-language search query for the dish/cuisine

    Returns:
        List of dicts with title, url, and a short content snippet —
        NOT structured ingredients like Spoonacular. The agent should
        read the snippet to judge relevance before recommending it.
    """

    results = tavily_client.search(query=query, max_results=5)
    return [
        {
            "name": r["title"],
            "source": r["url"],
            "snippet": r["content"][:300],
        }
        for r in results.get("results", [])
    ]


def get_recipe_information(recipe_id: int) -> dict | None:
    """
    Fetch full recipe details (ingredient quantities, step-by-step
    instructions, servings, ready time) from Spoonacular for a given
    recipe id. Returns None if unavailable — caller must handle that
    (e.g. web-sourced recipes with no id, or a failed lookup).
    """
    try:
        resp = requests.get(
            f"https://api.spoonacular.com/recipes/{recipe_id}/information",
            params={"apiKey": settings.SPOONACULAR_API_KEY, "includeNutrition": False},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "servings": data.get("servings"),
            "ready_in_minutes": data.get("readyInMinutes"),
            "ingredients": [
                f"{i['amount']} {i['unit']} {i['name']}".strip()
                for i in data.get("extendedIngredients", [])
            ],
            "instructions": data.get("instructions") or "",
        }
    except (requests.RequestException, KeyError):
        return None