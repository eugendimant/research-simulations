from pathlib import Path
from socsim.web.http import WebClient
from socsim.corpus.store import CorpusStore

def test_webclient_construct(tmp_path):
    c = WebClient(cache_path=tmp_path/"cache.sqlite")
    assert c.cache_path is not None

def test_corpus_store_roundtrip(tmp_path):
    p = tmp_path/"corpus.json"
    cs = CorpusStore.load(p)
    assert cs.units == []
    unit = {
        "id":"u1",
        "kind":"bibliography",
        "title":"t",
        "source":{"ref_type":"url","ref":"x","origin":"test","url":"x"},
        "tags":["metadata_only"],
        "payload":{"k":1},
        "provenance":{"added_by":"t","added_at_utc":"2020-01-01T00:00:00Z","extraction_method":"manual","notes":"n"},
    }
    cs.add_dict(unit, schema_path=Path("socsim/schema/atomic_unit_schema.json"))
    cs.save(p)
    cs2 = CorpusStore.load(p)
    assert len(cs2.units) == 1
