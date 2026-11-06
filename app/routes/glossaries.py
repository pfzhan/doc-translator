"""Routes of the glossary library."""

from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from .. import glossary, state
from ..glossaries import GlossaryError, GlossaryNotFound
from ..json_types import JSONDict

router = APIRouter()


@router.get("/api/glossaries")
def list_glossaries() -> list[JSONDict]:
    return state.glossaries.list()


@router.post("/api/glossaries")
def create_glossary(data: JSONDict = Body(...)) -> JSONDict:
    try:
        return state.glossaries.create(data)
    except GlossaryError as e:
        raise HTTPException(400, str(e)) from e


@router.put("/api/glossaries/{gid}")
def update_glossary(gid: str, data: JSONDict = Body(...)) -> JSONDict:
    try:
        return state.glossaries.update(gid, data)
    except GlossaryNotFound as e:
        raise HTTPException(404, str(e)) from e
    except GlossaryError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/api/glossaries/{gid}")
def delete_glossary(gid: str) -> JSONDict:
    try:
        state.glossaries.delete(gid)
    except GlossaryNotFound as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True}


@router.post("/api/glossaries/import")
def import_glossary(data: JSONDict = Body(...)) -> JSONDict:
    """Parse an uploaded CSV / TSV / "source = translation" file into entry text; nothing is saved until the user
    saves the glossary."""
    entries = glossary.parse_table(str(data.get("text") or ""))
    if not entries:
        raise HTTPException(400, "没有读到术语：每行应为“原文,译文”、“原文⇥译文”或“原文 = 译文”")
    return {"entries": glossary.render(entries), "count": len(entries)}


@router.get("/api/glossaries/{gid}/export")
def export_glossary(gid: str) -> Response:
    item = state.glossaries.get(gid)
    if not item:
        raise HTTPException(404, "术语表不存在")
    body = glossary.to_csv(glossary.parse(item["entries"]))
    name = quote(f"{item['name']}.csv")
    return Response(
        # The BOM lets Excel open the UTF-8 file with the right encoding
        "\ufeff" + body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"},
    )
