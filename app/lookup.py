"""Looking up a word or short phrase selected on a web page: a dictionary entry written by the model.

LLM engines ask for the entry as one JSON object (see prompts.LOOKUP_SYSTEM_PROMPT) and parse it here. Engines
without a model, such as Google, give a translation of the word as the only sense (see Translator.lookup).
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from .json_types import JSONDict
from .prompts import clean_output

# Senses kept from a reply; the prompt asks for at most 5, a model writing more is cut here
MAX_SENSES = 8


class Sense(BaseModel):
    pos: str = ""  # part of speech, abbreviated (n., v., adj., ...); empty when the model gave none
    meaning: str


class WordEntry(BaseModel):
    headword: str  # the selection in its dictionary form
    phonetic: str = ""  # pronunciation, such as IPA between slashes; empty when there is none
    senses: list[Sense] = Field(default_factory=list)
    # What the selection means in the sentence it was selected from; empty without a sentence
    in_context: str = ""

    @field_validator("senses")
    @classmethod
    def _cut_senses(cls, senses: list[Sense]) -> list[Sense]:
        return [s for s in senses if s.meaning.strip()][:MAX_SENSES]


_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def parse_entry(content: str, selection: str) -> WordEntry | None:
    """The entry in a model's reply; None when the reply holds no such JSON object, or one without any meaning.

    Models wrap the object in a code fence or a sentence now and then, so the outermost braces are taken. A missing
    headword is filled in with the selection itself."""
    found = _OBJECT_RE.search(clean_output(content))
    if not found:
        return None
    try:
        # Text from { to } that parses is an object
        data: JSONDict = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("headword"), str) or not data["headword"].strip():
        data["headword"] = selection
    try:
        entry = WordEntry.model_validate(data)
    except ValidationError:
        return None
    if not entry.senses and not entry.in_context.strip():
        return None
    return entry
