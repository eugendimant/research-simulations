from socsim.games.registry import make_game
from socsim.web.harvest import stable_bib_uid, normalize_doi

def test_registry_has_new_games():
    for name in ["repeated_trust","public_goods_punishment","discrete_choice","bdm"]:
        g = make_game(name)
        assert g.name == name

def test_doi_normalize():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert stable_bib_uid("t","10.1/x","crossref").startswith("bib_")
