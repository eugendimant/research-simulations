from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from .crossref import crossref_search, BibItem

# "Top 100" here means: 100 broad, high-frequency social-science constructs that appear across behavioral econ,
# social psych, marketing, and management, expressed as Crossref search queries.
#
# This list is intentionally broad to seed a metadata-only corpus. It does NOT create numeric effect sizes.
DEFAULT_QUERY_PACK_NEXT100: List[str] = [
    # Cooperation, norms, social preferences
    "dictator game meta-analysis",
    "ultimatum game meta-analysis",
    "trust game meta-analysis",
    "public goods game punishment meta-analysis",
    "public goods game reward meta-analysis",
    "prisoner's dilemma cooperation meta-analysis",
    "conditional cooperation public goods",
    "social norms descriptive injunctive meta-analysis",
    "norm salience field experiment",
    "social image incentives prosocial behavior",
    "reputation cooperation experiment",
    "inequity aversion experimental evidence",
    "reciprocity gift exchange experiment",
    "altruism warm glow experiment",
    "moral licensing experiment",
    "moral wiggle room experiment",
    "lying honesty experiment meta-analysis",
    "dishonesty meta-analysis",
    "cheating experiment die roll",
    "collusion experiment",
    # Discrimination, identity, social categories
    "taste-based discrimination experiment",
    "statistical discrimination experiment",
    "implicit bias IAT validity meta-analysis",
    "stereotype threat meta-analysis",
    "identity priming experiment",
    "minimal group paradigm meta-analysis",
    "ingroup bias economic games",
    "outgroup hostility experiment",
    "social identity theory experiment",
    "contact hypothesis meta-analysis",
    "dehumanization experiment",
    "political polarization intervention meta-analysis",
    "affective polarization experiment",
    # Beliefs, information, persuasion
    "belief elicitation incentivized experiment",
    "overconfidence experiment meta-analysis",
    "confirmation bias experiment",
    "motivated reasoning experiment",
    "misinformation correction experiment meta-analysis",
    "persuasion field experiment",
    "social influence conformity experiment meta-analysis",
    "peer effects field experiment",
    "network diffusion behavior change experiment",
    "trust in institutions survey experiment",
    # Risk, time, choice
    "risk preferences Holt Laur y experiment",
    "loss aversion experiment meta-analysis",
    "prospect theory experimental test",
    "time discounting experiment meta-analysis",
    "present bias field experiment",
    "delay discounting measurement validation",
    "ambiguity aversion experiment",
    "probability weighting experiment",
    "mental accounting experiment",
    # Labor, incentives, management
    "principal agent experiment effort",
    "performance pay field experiment meta-analysis",
    "intrinsic motivation crowding out experiment",
    "goal setting experiment meta-analysis",
    "feedback intervention meta-analysis",
    "monitoring and incentives experiment",
    "organizational culture experimentation A/B testing",
    # Marketing, consumer behavior
    "conjoint analysis choice-based conjoint",
    "willingness to pay experiment",
    "price framing experiment",
    "scarcity marketing experiment",
    "default effect experiment meta-analysis",
    "nudges meta-analysis",
    "choice overload experiment meta-analysis",
    "social proof marketing experiment",
    "anchoring experiment meta-analysis",
    "decoy effect attraction effect experiment",
    # Social preferences extensions
    "inequality and redistribution preferences experiment",
    "third-party punishment experiment meta-analysis",
    "norm enforcement experiment",
    "ultimatum responder rejection fairness",
    "trustworthiness reciprocity experiment",
    "cooperation reciprocity in repeated games",
    # Governance, corruption, crime
    "bribery game experiment",
    "corruption experiment meta-analysis",
    "tax compliance field experiment meta-analysis",
    "deterrence punishment certainty severity experiment",
    "collective action corruption experiment",
    # Public policy and interventions
    "energy conservation social norms field experiment",
    "charitable giving field experiment meta-analysis",
    "voter turnout social norms experiment",
    "vaccination message framing experiment",
    "public health behavior nudges meta-analysis",
    # Emotions, cognition, stress
    "time pressure decision making experiment",
    "cognitive load moral decision experiment",
    "emotion anger risk taking experiment",
    "stress cooperation experiment",
    # Methodology, measurement
    "measurement invariance survey cross-cultural",
    "item response theory graded response model",
    "acquiescence bias survey experiment",
    "social desirability bias measurement",
    "experimenter demand effects",
    "Hawthorne effect field experiment",
    "replication crisis psychology meta-analysis",
    "publication bias meta-analysis methods",
    "p-hacking evidence behavioral science",
    # Big constructs
    "trust scale validation",
    "big five personality economic preferences",
    "social value orientation measurement",
    "dark triad dishonesty experiment",
    "empathy prosocial behavior meta-analysis",
    "inequality aversion neural correlates",
    "norms expectations Bicchieri",
    "collective intelligence groups experiment",
    "team cooperation incentives experiment",
    "leadership trust experiment"
]

def expand_metadata_only(
    query_pack: List[str],
    total_target: int = 100,
    per_query_rows: int = 20,
    mailto: Optional[str] = None,
) -> List[Dict]:
    """Fetch Crossref metadata and return a list of schema-valid metadata-only units.
    Does NOT create numeric effect sizes.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    units: List[Dict] = []
    seen = set()

    for q in query_pack:
        items = crossref_search(q, rows=per_query_rows, mailto=mailto)
        for it in items:
            ref_type = "doi" if it.doi else "url"
            ref_val = it.doi or it.url or ""
            if not ref_val:
                continue
            key = (it.title or "", ref_val)
            if key in seen:
                continue
            seen.add(key)
            uid = f"lit_{abs(hash(key))%10**12:012d}"
            units.append({
                "id": uid,
                "title": it.title or q,
                "source": {"type": ref_type, "ref": ref_val},
                "domain_tags": ["bibliography", "metadata_only"],
                "applicability": {"games":["any"], "topics":["any"], "populations":["any"], "context_tags":["any"]},
                "payload": {"kind":"mechanism_reference", "effects":[
                    {"type":"reference_only","year": it.year, "container": it.container, "authors": it.authors, "query": q}
                ]},
                "provenance": {
                    "added_by":"socsim.autocorpus",
                    "added_at_utc": now,
                    "extraction_method":"programmatic_metadata_only",
                    "notes":"Metadata-only import via Crossref; no effect sizes extracted."
                }
            })
            if len(units) >= total_target:
                return units
    return units
