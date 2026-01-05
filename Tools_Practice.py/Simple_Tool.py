from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query."""
    return f"Found {limit} results for '{query}'"


@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)   # -> web_search


@tool("Calculator")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

print(calc.invoke("2 + 5 * 3"))
res = search.invoke('what is the capital of india')
print(res)

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description='City name or coordinates')
    units : Literal['celsius', 'fahrenheit'] = Field(
        default='celsius',
        description='Temperature unit preference'
    )
    inclue_forecast: bool = Field(
        default=False,
        description='Include 5-day forecast'
    )
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = 'celsius', include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp=22 if units == 'celsius' else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


