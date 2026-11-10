"""Looking up a word or short phrase selected on a web page: a dictionary entry written by an LLM.

The model writes the entry as one JSON object (see prompts.LOOKUP_SYSTEM_PROMPT), parsed here. Like selected text
that is translated, the entry is written in the target language, or in the second language when the word itself is
in the target language; the model tells which, since a detector cannot tell the language of a single word.
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
    phonetic: str = ""  # pronunciation, such as IPA between slashes or pinyin; empty when there is none
    senses: list[Sense] = Field(default_factory=list)
    # What the selection means in the sentence it was selected from; empty without a sentence
    in_context: str = ""
    lang: str = ""  # the language the entry is written in; set by the engine, not the model

    @field_validator("senses")
    @classmethod
    def _cut_senses(cls, senses: list[Sense]) -> list[Sense]:
        return [s for s in senses if s.meaning.strip()][:MAX_SENSES]


class ModelEntry(WordEntry):
    """The entry as the model writes it: also whether the selection is in the target language, in which case the
    entry is written in the second language."""

    in_target_language: bool = False

    def written_in(self, target_lang: str, alt_lang: str) -> WordEntry:
        lang = alt_lang if self.in_target_language and alt_lang else target_lang
        return WordEntry(**self.model_dump(exclude={"in_target_language", "lang"}), lang=lang)


_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def parse_entry(content: str, selection: str) -> ModelEntry | None:
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
    data.pop("lang", None)  # not the model's to set
    try:
        entry = ModelEntry.model_validate(data)
    except ValidationError:
        return None
    if not entry.senses and not entry.in_context.strip():
        return None
    return entry
