import asyncio
import json

import httpx

from app.glossary import candidates, merge, parse, relevant
from app.runner import Cache, Runner
from app.translators import OpenAITranslator

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
    # cube 只出现在 data cube 里，不单独成条；虚词和只出现一次的词不进来
    assert "cube" not in found and "the greedy" not in found
    assert "Summary statistics" not in found and "summary statistics" not in found


def test_possessives_do_not_leave_word_fragments():
    """撇号不是断句标点：view's set 不能切出 “s set”，don't 也不能切出 “t …”。"""
    text = ["The view's set of output columns must cover the query's set of output columns, and we don't drop columns."] * 4
    found = candidates(text)
    assert "output column" in found or "output columns" in found
    assert not any(term.split()[0] in ("s", "t") for term in found), found


def test_a_word_repeated_across_table_cells_is_not_a_phrase():
    """表格里相邻两格都是 semi-supervised，拼成的 “semi-supervised semi-supervised” 不是术语。"""
    rows = ["Vinyals & Kaiser el al. (2014) semi-supervised semi-supervised 92.1"] * 4
    assert not any(len(set(term.casefold().split())) < len(term.split()) for term in candidates(rows))


def test_candidates_skip_text_without_spaces_between_words():
    assert candidates(["数据立方体的实现。数据立方体很大。"] * 10) == []


def test_relevant_entries_follow_the_batch_text():
    entries = [("query", "查询"), ("data cube", "数据立方体"), ("SQL", "SQL"), ("view", "视图")]
    assert relevant(entries, ["Two queries over data  cubes"]) == [("query", "查询"), ("data cube", "数据立方体")]
    # 缩写分大小写，词中间的 view 不算
    assert relevant(entries, ["no sql in this review"]) == []


def test_relevant_entries_see_through_style_marks():
    entries = [("word types", "词型")]
    assert relevant(entries, ["to be the same word{b} type{/b}?"]) == [("word types", "词型")]


def test_user_entries_win_over_found_ones():
    user = parse("Data Cube = 数据立方\nnot an entry\n = empty")
    assert user == [("Data Cube", "数据立方")]
    assert merge(user, [("data cube", "数据立方体"), ("view", "视图")]) == [("Data Cube", "数据立方"), ("view", "视图")]


def _translator(requests, terms_reply="data cube = 数据立方体\ngreedy algorithm = 贪心算法\nOLAP = OLAP\nnot asked = 无"):
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

    tr = OpenAITranslator("zh-CN", api_key="k", base_url="https://api.example.com/v1", model="m",
                          transport=httpx.MockTransport(handler), max_items=2)
    tr.service_id = "svc"
    return tr


def test_terms_are_defined_once_and_sent_only_where_they_appear(tmp_path):
    requests = []
    tr = _translator(requests)
    runner = Runner(tr, cache=Cache(tmp_path / "c.sqlite3"), skip_same_lang=False)
    asyncio.run(runner.translate_all(DOC))
    term_requests = [user for system, user in requests if "terminology list" in system]
    assert len(term_requests) == 1 and "data cube" in term_requests[0]
    # 候选是按词频数出来的：提示词要让模型删掉截出来的片段、公式变量和宽泛的普通词
    term_system = next(system for system, _user in requests if "terminology list" in system)
    assert "cut out of a longer phrase" in term_system and "variable names" in term_system
    assert "everyday words" in term_system
    # 模型多给的、不在候选里的条目不收
    assert tr.doc_glossary == [("data cube", "数据立方体"), ("greedy algorithm", "贪心算法"), ("OLAP", "OLAP")]
    assert runner.info["terms"] == 3
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
    """同一份文件，提取规则改了以后候选清单跟着变：仍用第一次定下的术语表，
    没变的段落照样命中缓存，不会整篇重翻。"""
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
    # 只有新加的那一段去请求翻译
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
    """清理一份文件的缓存后重新翻译：术语表和段落都登记在这份文件名下，一起清掉，重新请求。"""
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
    """只排版时缺术语表不去请求，按没有术语表的缓存键找，旧记录照样能补生成。"""
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
    # language model、model 多半出现在 large language model 里；language 单独出现得多，留下
    assert _drop_covered(terms, counts) == ["language", "large language model"]
