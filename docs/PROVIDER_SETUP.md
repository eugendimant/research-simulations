# Getting free AI keys — step by step

This is the five-minute setup that switches on **AI-written open-ended text**.

You do not have to do all of it. **Google AI Studio alone (step 1) restores most
of the value** and takes about ninety seconds. Everything after it is redundancy:
each extra provider is one more fallback when the one before it is rate-limited.

**Nothing here is required to use the app.** With no keys at all, open-ended text
comes from the built-in engine, every run completes, and numeric data is
identical either way. Keys change only whether open-text answers are written by
an LLM.

---

## Before you start: two rules

1. **Paste each key into your password manager the moment you create it.**
   Providers show a new key exactly once. Streamlit will not show you the saved
   value again either. A key you did not save is a key you must recreate.
2. **Never put a key in the repository**, not even XOR-encoded or inside a
   comment. This repository is public: anything committed is readable by
   anyone, and stays readable in git history after it is deleted. GitHub's
   secret-scanning partner programme also notifies several of these providers
   on detection, and some revoke the key automatically. `tests/test_no_secrets_in_repo.py`
   fails the build if a key-shaped string appears in a tracked file.

> **If you used this tool before October 2026:** the keys that were previously
> committed to this repository should be treated as compromised and revoked at
> each provider, then replaced with fresh ones created below. They were
> publicly readable, and several may already have been auto-revoked.

---

## Step 1 — Google AI Studio (Gemini). Start here.

The most generous free tier, no credit card, and it powers three of the nine
entries in the provider chain.

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with a Google account.
3. Click **Create API key**.
4. If asked to pick a project, choose any existing one or **Create project**.
5. Copy the key (it starts with `AIzaSy`) and save it in your password manager.

Secret name to use: **`GOOGLE_API_KEY`**

Free tier: Gemini Flash Lite at 30 requests/minute, Gemini Flash at 15
requests/minute, 1M tokens/minute. No card required.

Caveats: limits are per project, not per key, so a second key in the same
project does not double anything. AI Studio's free tier is not offered in every
country; if the key page refuses to issue one, that is a region block rather
than a fault, and the next five providers still work.

---

## Step 2 — Groq

Very high daily volume, and the fastest responses in the chain.

1. Go to **https://console.groq.com/keys**
2. Sign in (Google or GitHub works).
3. Click **Create API Key**, give it any name, click **Submit**.
4. Copy the key (starts with `gsk_`) and save it. Groq will not show it again.

Secret name: **`GROQ_API_KEY`**

Free tier: ~30 requests/minute, ~14,400 requests/day. No card required.

Caveat: the key is shown exactly once, on creation. There is no way to read it
back later — if it is not in your password manager, create a new one.

---

## Step 3 — SambaNova Cloud

1. Go to **https://cloud.sambanova.ai/apis**
2. Sign up / sign in.
3. Open the **APIs** section and click **Generate new key**.
4. Copy the key and save it.

Secret name: **`SAMBANOVA_API_KEY`**

Free tier: persistent free tier, ~20 requests/minute.

Caveat: sign-up requires email verification before the key page will issue
anything, and the free allowance is credit-based, so it can run out rather than
merely rate-limit.

---

## Step 4 — OpenRouter

Last resort in the chain; useful because it fronts several free models.

1. Go to **https://openrouter.ai/keys**
2. Sign in (Google or GitHub).
3. Click **Create Key**, name it, confirm.
4. Copy the key (starts with `sk-or-v1-`) and save it.

Secret name: **`OPENROUTER_API_KEY`**

Caveats: only models whose name ends in `:free` cost nothing, and their daily
cap is low (and lower still on an account that has never had credit on it). A
`402` from OpenRouter means the model was not a free one, not that the key is
bad.

---

## Where to paste the keys

Keys live **only** in the deployment environment. There is no key file in this
repository, by design.

### Streamlit Community Cloud (how this app is deployed)

1. Open **https://share.streamlit.io** and select the app.
2. Click **⋮ → Settings → Secrets**.
3. Paste the block below, with your real keys filled in. Delete any line you
   do not have a key for — a missing line is fine, an empty one is too.
4. Click **Save**. Streamlit restarts the app and picks the keys up.

```toml
GOOGLE_API_KEY = "AIzaSy..."
GROQ_API_KEY = "gsk_..."
SAMBANOVA_API_KEY = "..."
OPENROUTER_API_KEY = "sk-or-v1-..."
```

The same block, with placeholders and comments, is in
[`secrets.toml.example`](../secrets.toml.example) at the repository root —
copy from there.

### Running locally

Either copy `secrets.toml.example` to `.streamlit/secrets.toml` (that path is
git-ignored, so it cannot be committed by accident) and fill it in, or export
the same names as environment variables:

```bash
export GOOGLE_API_KEY="AIzaSy..."
streamlit run simulation_app/app.py
```

Environment variables and `st.secrets` are read identically and are
interchangeable — `tests/test_builtin_provider_config.py` pins that down for
every provider slot.

---

## Check that it worked

1. Open the app with `?admin=1` and sign in with `ADMIN_PASSWORD`.
2. Go to the **LLM** tab.
3. **Built-in Provider Keys** shows which slots the deployment can see. A key
   you just saved should read **Configured: Yes**.
4. Click **Test providers now**. This sends one tiny request per configured
   provider and reports each one as:
   - **✅ OK** — the key works; latency is shown.
   - **❌ Failed** — the key is present but was rejected, rate-limited, or the
     model is unavailable. The reason is shown. A key that was once public may
     have been auto-revoked; create a fresh one.
   - **— Not configured** — no key for that slot; it is not contacted at all.

Key material is never displayed, logged, or included in any error text shown
here: failure reasons are scrubbed of every configured key before display.

If every provider fails, nothing breaks — open-ended text comes from the
built-in engine and runs still complete.

---

## Rotating a key

1. Create the new key at the provider, following the step above for it.
2. Paste it over the old value in **Settings → Secrets** and save.
3. Delete the old key in the provider's console.
4. Click **Test providers now** to confirm the new one answers.

Keys are never stored anywhere else, so there is nothing else to clean up.

---

## The rules that keep this from breaking again

Keys belong only in the deployment and in your password manager; never in the
repository, a commit message, an issue or a chat. Never drop a provider from
the chain without adding one in its place, and run **Test providers now** after
any change to the LLM code. The full policy, and the list of tests that enforce
each rule in CI, is in [`KEY_POLICY.md`](KEY_POLICY.md).
