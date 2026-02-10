# Agent Change Log

## 2026-02-10 — v1.9.1
### Bug Fixes
- Fixed analytics dashboard NameError on `clean_scales` — recovered from session state
- Fixed question removal (X button) — session state flag approach replaces unreliable in-render-loop removal
- Fixed live data preview — now shows all columns/scales (removed 5-scale and 2-OE limits)
- Fixed title truncation — full title display via HTML rendering (was truncated at 30 chars)
- Fixed treatment logic — unmatched blocks in BlockRandomizer no longer default to visible for all conditions

### LLM Pipeline (Critical)
- Improved error diagnostics: HTTP status codes, auth errors (401/403), rate limits (429), connection errors logged explicitly
- Added 5-provider failover chain: Groq → Cerebras → Together AI → SambaNova → OpenRouter
- Added 8 fallback response parsing strategies (truncated JSON recovery, numbered lists, newline-delimited)
- Added retry with exponential backoff (up to 3 retries per call per provider)
- Added cooldown-based auto-recovery for temporarily disabled providers
- Engine prefill gate no longer requires `is_llm_available` — always attempts LLM generation
- Providers reset after failed prefill so on-demand generation gets a fresh start

### Response Realism (4 Deep Iterations)
- Within-person coherence (g-factor) with differential construct-type loadings (Podsakoff et al., 2003)
- Personality × Condition interaction effects (ELM, Petty & Cacioppo, 1986; satisficing, Krosnick, 1991)
- Response time simulation with log-normal distributions (Callegaro et al., 2015; Yan & Tourangeau, 2008)
- Correlated response styles (acquiescence-extremity-social desirability) via Cholesky decomposition (Baumgartner & Steenkamp, 2001)

### UX Improvements
- Redesigned landing page: replaced 4 expanders with professional tabbed design
- Analytics dashboard: added prominent dark gradient header with icon
- Report: updated provider display names for Together AI and SambaNova

### Research Domain Detection
- Added 50+ missing keyword sets across research domains
- Expanded matching with better fallback logic

## 2025-01-27
- Updated landing page branding, attribution, and method summary download link.
- Added Qualtrics survey PDF upload to improve domain inference and stored excerpts in metadata.
- Created a new methods summary document outlining persona modeling and recommended uploads.
