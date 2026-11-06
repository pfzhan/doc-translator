"""Types for JSON data the server reads and returns."""

from typing import Any

# A JSON object: a request body or an API answer. Values are mixed JSON (numbers, strings, lists, objects), hence Any
JSONDict = dict[str, Any]
