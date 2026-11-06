import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import job_control, state
from app.glossaries import GlossaryError, GlossaryNotFound, GlossaryStore
from app.glossary import candidates, merge, parse, parse_table, relevant, render, to_csv
from app.runner import Cache, Runner
from app.translators import OpenAITranslator, create_translator
from tests.test_history import submit, wait_done

DOC = [
    "Implementing Data Cubes Efficiently",
    "Decision support systems query a data cube built over a warehouse.",
    "Each cell of the data cube is a view; OLAP tools read the data cube directly.",
    "The greedy algorithm picks views to materialize one at a time.",
    "We show that the greedy algorithm is close to optimal for the data cube.",
    "OLAP vendors precompute views, and the greedy algorithm tells them which ones.",
    "A lattice describes how each view depends on the others.",
    "Summary statistics are cheap.",
]


def test_candidates_keep_repeated_phrases_and_acronyms():
    found = candidates(DOC)
    assert "data cube" in found
    assert "greedy algorithm" in found
    assert "OLAP" in found
    # cube occurs only within data cube and gets no entry of its own; function words and words occurring once are left
    # out
    assert "cube" not in found and "the greedy" not in found
    assert "Summary statistics" not in found and "summary statistics" not in found


def test_possessives_do_not_leave_word_fragments():
    """An apostrophe does not split phrases: view's set must not yield “s set”, nor don't yield “t …”."""
    text = [
        "The view's set of output columns must cover the query's set of output columns, and we don't drop columns."
    ] * 4
    found = candidates(text)
    assert "output column" in found or "output columns" in found
    assert not any(term.split()[0] in ("s", "t") for term in found), found


def test_a_word_repeated_across_table_cells_is_not_a_phrase():
    """Two adjacent table cells both reading semi-supervised do not make “semi-supervised semi-supervised” a term."""
    rows = ["Vinyals & Kaiser el al. (2014) semi-supervised semi-supervised 92.1"] * 4
    assert not any(len(set(term.casefold().split())) < len(term.split()) for term in candidates(rows))


def test_candidates_skip_text_without_spaces_between_words():
    assert candidates(["数据立方体的实现。数据立方体很大。"] * 10) == []


def test_relevant_entries_follow_the_batch_text():
    entries = [("query", "查询"), ("data cube", "数据立方体"), ("SQL", "SQL"), ("view", "视图")]
    assert relevant(entries, ["Two queries over data  cubes"]) == [("query", "查询"), ("data cube", "数据立方体")]
    # Acronyms are case-sensitive, and view inside a word does not count
    assert relevant(entries, ["no sql in this review"]) == []


def test_relevant_entries_see_through_style_marks():
    entries = [("word types", "词型")]
    assert relevant(entries, ["to be the same word{b} type{/b}?"]) == [("word types", "词型")]


def test_user_entries_win_over_found_ones():
    user = parse("Data Cube = 数据立方\nnot an entry\n = empty")
    assert user == [("Data Cube", "数据立方")]
    assert merge(user, [("data cube", "数据立方体"), ("view", "视图")]) == [("Data Cube", "数据立方"), ("view", "视图")]


def _translator(
    requests, terms_reply="data cube = 数据立方体\ngreedy algorithm = 贪心算法\nOLAP = OLAP\nnot asked = 无"
):
    async def handler(request):
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        user = body["messages"][-1]["content"]
        requests.append((system, user))
        if "terminology list" in system:
            content = terms_reply
        elif "[[source_end]]" in user:
            items = [line for line in user.splitlines() if line.startswith("[[p")]
            content = "\n".join(f"{item}\n译文" for item in items)
        else:
            content = "译文"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = OpenAITranslator(
        "zh-CN",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        max_items=2,
    )
    tr.service_id = "svc"
    return tr


def test_terms_are_defined_once_and_sent_only_where_they_appear(tmp_path):
    requests = []
    tr = _translator(requests)
    runner = Runner(tr, cache=Cache(tmp_path / "c.sqlite3"), skip_same_lang=False)
    asyncio.run(runner.translate_all(DOC))
    term_requests = [user for system, user in requests if "terminology list" in system]
    assert len(term_requests) == 1 and "data cube" in term_requests[0]
    # Candidates come from word counts, so the prompt must ask the model to drop cut-off fragments, formula variables
    # and broad common words
    term_system = next(system for system, _user in requests if "terminology list" in system)
    assert "cut out of a longer phrase" in term_system and "variable names" in term_system
    assert "everyday words" in term_system
    # Extra entries from the model that were not candidates are not accepted
    assert tr.doc_glossary == [("data cube", "数据立方体"), ("greedy algorithm", "贪心算法"), ("OLAP", "OLAP")]
    assert runner.info.terms == 3
    batches = [(system, user) for system, user in requests if "terminology list" not in system]
    for system, user in batches:
        assert ("data cube = 数据立方体" in system) == ("data cube" in user)
        assert ("OLAP = OLAP" in system) == ("OLAP" in user)


def test_resuming_reuses_the_same_glossary_and_cache(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    first = []
    asyncio.run(Runner(_translator(first), cache=cache, skip_same_lang=False).translate_all(DOC))
    again = []
    tr = _translator(again)
    out = asyncio.run(Runner(tr, cache=cache, skip_same_lang=False).translate_all(DOC))
    assert again == []
    assert tr.doc_glossary and set(out) == {"译文"}


def test_the_same_file_keeps_its_glossary_when_extraction_changes(tmp_path):
    """For the same file, the candidate list changes when extraction rules change, but the glossary fixed the first
    time is still used, so unchanged paragraphs still hit the cache and the document is not retranslated."""
    cache = Cache(tmp_path / "c.sqlite3")
    first = Runner(_translator([]), cache=cache, skip_same_lang=False)
    first.set_document("sha-of-file")
    asyncio.run(first.translate_all(DOC))

    changed = DOC + ["A new paragraph mentions the lattice and the lattice again, the lattice."]
    requests = []
    tr = _translator(requests)
    again = Runner(tr, cache=cache, skip_same_lang=False)
    again.set_document("sha-of-file")
    asyncio.run(again.translate_all(changed))
    assert not any("terminology list" in system for system, _user in requests)
    assert tr.doc_glossary == first.translator.doc_glossary
    # Only the newly added paragraph is requested
    sent = [user for system, user in requests]
    assert len(sent) == 1 and "A new paragraph" in sent[0]


def test_translate_file_names_the_document(tmp_path):
    from app.formats import translate_file

    src = tmp_path / "a.md"
    src.write_text("\n\n".join(DOC), encoding="utf-8")
    runner = Runner(_translator([]), cache=Cache(tmp_path / "c.sqlite3"), skip_same_lang=False)
    asyncio.run(translate_file(src, tmp_path, runner, False, "zh-CN"))
    assert runner.document and len(runner.document) == 40


def test_clearing_a_document_defines_terms_and_translates_again(tmp_path):
    """Retranslating after clearing a file's cache: the glossary and paragraphs are registered under this file, are
    cleared together and requested again."""
    cache = Cache(tmp_path / "c.sqlite3")
    first = Runner(_translator([]), cache=cache, skip_same_lang=False)
    first.set_document("doc")
    asyncio.run(first.translate_all(DOC))
    cache.purge("doc", cache.owned("doc"))
    requests = []
    again = Runner(_translator(requests), cache=cache, skip_same_lang=False)
    again.set_document("doc")
    asyncio.run(again.translate_all(DOC))
    assert any("terminology list" in system for system, _user in requests)
    assert sum(user.count("[[p") for system, user in requests if "terminology list" not in system) == len(DOC)


def test_relayout_only_never_asks_for_terms(tmp_path):
    """When only typesetting, a missing glossary is not requested; lookup uses cache keys without a glossary, so old
    records can still regenerate outputs."""
    cache = Cache(tmp_path / "c.sqlite3")
    plain = _translator([])
    plain.auto_glossary = False
    asyncio.run(Runner(plain, cache=cache, skip_same_lang=False).translate_all(DOC))
    requests = []
    tr = _translator(requests)
    out = asyncio.run(Runner(tr, cache=cache, skip_same_lang=False, cache_only=True).translate_all(DOC))
    assert requests == [] and tr.doc_glossary == []
    assert set(out) == {"译文"}


def test_loose_user_lines_go_with_every_batch():
    tr = _translator([])
    tr.glossary = "Keep product names in English\nview = 视图"
    assert tr.glossary_for(["a lattice"]) == "Keep product names in English"
    assert tr.glossary_for(["each view"]) == "Keep product names in English\nview = 视图"


def test_covered_terms_are_found_anywhere_inside_longer_ones():
    from collections import Counter

    from app.glossary import _drop_covered

    counts = Counter({"large language model": 10, "language model": 11, "language": 30, "model": 12})
    terms = ["language", "model", "language model", "large language model"]
    # language model and model mostly occur within large language model; language occurs often on its own and stays
    assert _drop_covered(terms, counts) == ["language", "large language model"]


def test_parse_table_reads_csv_tsv_and_equals_lines():
    assert parse_table('原文,译文\ndata cube,数据立方体\n"a, b",甲乙\n') == [
        ("data cube", "数据立方体"),
        ("a, b", "甲乙"),
    ]
    assert parse_table("﻿Source\tTarget\nOLAP\t联机分析处理\n") == [("OLAP", "联机分析处理")]
    assert parse_table("OLAP = 联机分析处理\nview = 视图") == [("OLAP", "联机分析处理"), ("view", "视图")]
    # The format is decided from all lines: a comma in the first term does not make the file CSV
    assert parse_table("Smith, J. = 史密斯\nview = 视图") == [("Smith, J.", "史密斯"), ("view", "视图")]
    # A first line that is an ordinary entry is not mistaken for a header; rows with an empty side are dropped
    assert parse_table("term,术语\nx,\n") == []
    assert parse_table("view,视图\nx,\n") == [("view", "视图")]
    assert parse_table("") == []


def test_csv_export_imports_back():
    entries = [("a, b", '甲"乙"'), ("view", "视图")]
    assert parse_table(to_csv(entries)) == entries


def test_terms_with_an_equals_sign_survive_saving():
    assert parse("view=视图\nE=mc2 = 质能方程") == [("view", "视图"), ("E=mc2", "质能方程")]
    entries = parse_table("E=mc2,质能方程\nview,视图")
    assert entries == [("E=mc2", "质能方程"), ("view", "视图")] and parse(render(entries)) == entries


def test_glossary_store_saves_only_entries(tmp_path):
    path = tmp_path / "glossaries.json"
    store = GlossaryStore(path)
    item = store.create({"name": " 数据库 ", "entries": "view = 视图\nnot an entry\n= x\nOLAP = 联机分析处理"})
    assert item["name"] == "数据库" and item["entries"] == "view = 视图\nOLAP = 联机分析处理"
    assert store.list()[0]["count"] == 2
    store.update(item["id"], {"name": "数据库 2", "entries": "view = 视图"})
    assert GlossaryStore(path).get(item["id"])["name"] == "数据库 2"
    with pytest.raises(GlossaryError):
        store.create({"name": "  "})
    with pytest.raises(GlossaryNotFound):
        store.update("nope", {"name": "x"})
    store.delete(item["id"])
    assert GlossaryStore(path).list() == []
    with pytest.raises(GlossaryNotFound):
        store.delete(item["id"])


def test_entries_text_keeps_order_and_skips_deleted(tmp_path):
    store = GlossaryStore(tmp_path / "glossaries.json")
    a = store.create({"name": "a", "entries": "view = 视图"})["id"]
    b = store.create({"name": "b", "entries": "cube = 立方体"})["id"]
    assert store.entries_text([b, "gone", a]) == "cube = 立方体\nview = 视图"
    assert store.entries_text([]) == ""


def test_glossary_api_and_job_selection(app_env):
    main = app_env
    with TestClient(main.app) as client:
        made = client.post("/api/glossaries", json={"name": "法学", "entries": "Book = 卷"}).json()
        assert client.get("/api/glossaries").json()[0]["count"] == 1
        assert (
            client.put(f"/api/glossaries/{made['id']}", json={"name": "法学", "entries": "Book = 篇"}).status_code
            == 200
        )
        assert client.post("/api/glossaries", json={"name": ""}).status_code == 400
        assert client.put("/api/glossaries/nope", json={"name": "x"}).status_code == 404

        imported = client.post("/api/glossaries/import", json={"text": "term,术语\nBook,卷\n"}).json()
        assert imported == {"entries": "Book = 卷", "count": 1}
        assert client.post("/api/glossaries/import", json={"text": "nothing here"}).status_code == 400

        export = client.get(f"/api/glossaries/{made['id']}/export")
        assert export.status_code == 200 and "attachment" in export.headers["content-disposition"]
        assert export.content.decode("utf-8-sig").splitlines() == ["原文,译文", "Book,篇"]

        # Ids of glossaries that do not exist are dropped when the job is submitted
        form = {"service_id": "mock", "target_lang": "zh-CN", "glossary_ids": [made["id"], "gone"]}
        r = client.post("/api/jobs", files={"file": ("a.md", b"# T\n\nSome text here.\n")}, data=form)
        assert r.status_code == 200 and r.json()["glossary_ids"] == [made["id"]]

        assert client.delete(f"/api/glossaries/{made['id']}").json() == {"ok": True}
        assert client.delete(f"/api/glossaries/{made['id']}").status_code == 404


def test_job_glossaries_follow_the_services_own_entries():
    service = {"id": "s", "glossary": "OLAP = 联机分析"}
    assert job_control._with_glossaries(service, "view = 视图")["glossary"] == "OLAP = 联机分析\nview = 视图"
    assert job_control._with_glossaries({"id": "s"}, "view = 视图")["glossary"] == "view = 视图"
    # Without a glossary the service is returned unchanged, so the cache key of earlier records stays the same
    assert job_control._with_glossaries(service, "") is service


def test_job_keeps_the_glossary_entries_its_translations_were_made_with(app_env, monkeypatch):
    used = []

    def recording(service, *args):
        used.append(service.get("glossary", ""))
        return create_translator(service, *args)

    monkeypatch.setattr(job_control, "create_translator", recording)
    gid = state.glossaries.create({"name": "g", "entries": "view = 视图"})["id"]
    with TestClient(app_env.app) as client:
        job = wait_done(client, submit(client, name="a.md", glossary_ids=gid)["id"])
        assert used == ["view = 视图"]
        state.glossaries.update(gid, {"name": "g", "entries": "view = 视界"})
        # The other version and cache clearing compute the keys of the translations already made
        assert client.post(f"/api/jobs/{job['id']}/variant").status_code == 200
        assert client.get(f"/api/jobs/{job['id']}/cache").status_code == 200
        assert used[1:] == ["view = 视图", "view = 视图"]
        # Translating again uses the edited glossary, and later keys follow it
        wait_done(client, client.post(f"/api/jobs/{job['id']}/retry").json()["id"])
        assert state.jobs.get(job["id"]).glossary_text == "view = 视界"
        state.glossaries.delete(gid)
        assert client.get(f"/api/jobs/{job['id']}/cache").status_code == 200
        # Continuing checks the configuration, the job creates its engine when it starts, then the cache is read
        assert used[3:] == ["view = 视界", "view = 视界", "view = 视界"]
