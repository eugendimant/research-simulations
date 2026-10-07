# Deployment secrets

The repository never contains API keys or passwords. Everything below is read at
runtime from environment variables or, on Streamlit Community Cloud, from
`st.secrets`. A secret that is not set simply disables the feature it powers —
the app still loads and still generates data.

## Built-in AI (free LLM providers)

The **Built-in AI** generation method calls free-tier LLM providers to write
open-ended text. Each provider slot takes one key. Set **at least one**; the
generator tries them in the order below and falls through to the next when one
rate-limits or fails.

| Order | Provider | Secret name | Where to get a free key |
|-------|----------|-------------|-------------------------|
| 1 | Google AI Studio (Gemini) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | https://aistudio.google.com/apikey |
| 2 | Groq | `GROQ_API_KEY` | https://console.groq.com/keys |
| 3 | SambaNova Cloud | `SAMBANOVA_API_KEY` | https://cloud.sambanova.ai/apis |
| 4 | OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |

**Step-by-step instructions** — which page to open, what to click, which plan to
pick, and where to paste the result — are in
[docs/PROVIDER_SETUP.md](PROVIDER_SETUP.md). A fill-in-the-blanks TOML template
is at [`secrets.toml.example`](../secrets.toml.example).

### These secrets are optional

The app is fully usable with none of them set. Built-in AI tries every
configured provider in order and takes whatever it cannot get from them from
the built-in (non-LLM) text engine instead, so a run always completes: no error
banner, no disabled button, a complete dataset with an open-ended response for
every participant. After generation the data-source breakdown reports exactly
what came from where.

What the keys change is only whether open-ended text is **AI-written** or
**engine-written**. Numeric data is never affected by them.

Users can also paste their own key in the app ("AI (your API key)") without any
deployment secret being set. Unlike the deployment secrets, that path reports an
error when the key does not work — someone who supplies their own key needs to
know it is failing rather than silently receiving engine-written text.

## There is no key file in this repository

Keys live only in the deployment environment. The optional
`simulation_app/utils/builtin_free_keys.py` module that v1.2.9.1 supported was
removed in v1.2.9.3, along with the loader that read it.

The reason is simple: a key committed to a repository is readable by everyone
who can read the repository, and remains readable in git history after the file
is deleted. Obfuscation does not change that — XOR encoding is encoding, not
protection. GitHub's secret-scanning partner programme also notifies several of
these providers on detection, and some revoke the key automatically, which
switches AI text off without warning.

`tests/test_no_secrets_in_repo.py` enforces this: it fails the build if a
key-shaped string appears in any tracked file, if a file named like the removed
key module reappears, or if an XOR-decoded byte block shows up.

This does mean someone who forks the repository must supply their own key to
get AI-written text. That is the intended trade: the fork still works — it
falls back to the built-in text engine and generates complete datasets — and
[docs/PROVIDER_SETUP.md](PROVIDER_SETUP.md) walks through getting a free key in
about five minutes.

## Password-gated pages

| Secret name | Gates |
|-------------|-------|
| `ADMIN_PASSWORD` or `ADMIN_PASSWORD_SHA256` | the `?admin=1` diagnostics page |
| `ANALYTICS_DASHBOARD_PASSWORD` | the analytics dashboard |

With neither set, the corresponding page stays locked. This is the intended
default: no secret, no access.

## Instructor and student email (optional)

When a run finishes, the app can email the instructor the generated dataset, the
analysis and the student package. This needs an SMTP account. With the SMTP
secrets unset the app still works; the notification is skipped and the admin page
(`?admin=1`, tab **Email Delivery**) says so.

| Secret name | Meaning | Default |
|-------------|---------|---------|
| `SMTP_SERVER`, `SMTP_USERNAME`, `SMTP_PASSWORD` | the sending account (all three are required) | unset |
| `SMTP_PORT` | port | `587` |
| `SMTP_USE_TLS` | STARTTLS on the connection | `true` |
| `SMTP_FROM_EMAIL`, `SMTP_FROM_NAME` | the From header | the username, "Behavioral Experiment Simulation Tool" |
| `INSTRUCTOR_NOTIFICATION_EMAIL` | who receives the instructor mail; **several addresses may be listed, separated by commas** | the address built into the app |
| `INSTRUCTOR_EMAIL_MODE` | `split` sends a summary mail with the analysis in the body, then a threaded second mail with the attachments; `single` sends one mail | `split` |
| `EMAIL_MAX_MESSAGE_MB` | largest message the app will send (the app shrinks or drops attachments to fit) | `15` |
| `INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR`, `INSTRUCTOR_EMAIL_MAX_PER_DAY` | caps on instructor mail so the mailbox cannot be flooded | `12` per session per hour, `200` per day |
| `USER_EMAIL_MAX_PER_SESSION_PER_HOUR`, `USER_EMAIL_MAX_PER_HOUR`, `USER_EMAIL_MAX_PER_DAY` | caps on mail triggered by students (their own copy of the package) | `5`, `60`, `100` |

Every send attempt is logged (one masked `EMAIL-DELIVERY` line in the app log and
`data/email_delivery_log.jsonl`) and shown on the admin page, which also has a
test-email button with selectable content (body only, `.md`, `.html`, `.zip`,
3 MB, 10 MB) so you can find out what your mail system holds back.

Two things help when mail is accepted by the SMTP server but never shows up:

- **List a second recipient outside your main mail system**, for example
  `INSTRUCTOR_NOTIFICATION_EMAIL = "you@university.edu, you@gmail.com"`. If one
  inbox receives the mail and the other does not, the filter is on the receiving
  side.
- **Use a From address that matches the sending account.** A From address on the
  recipient's own domain sent through a different server is treated as spoofing
  by Microsoft 365 and is quietly junked or quarantined. The admin tab warns
  about this. Search Junk and the Microsoft 365 quarantine for the Message-ID the
  tab prints.

## Setting them on Streamlit Community Cloud

1. Open the app at https://share.streamlit.io and pick it.
2. **⋮ → Settings → Secrets**.
3. Paste the keys in TOML form, one per line:

   ```toml
   GOOGLE_API_KEY = "..."
   GROQ_API_KEY = "..."
   SAMBANOVA_API_KEY = "..."
   OPENROUTER_API_KEY = "..."
   ADMIN_PASSWORD = "..."
   ANALYTICS_DASHBOARD_PASSWORD = "..."
   # optional, for the instructor email (see above)
   SMTP_SERVER = "..."
   SMTP_USERNAME = "..."
   SMTP_PASSWORD = "..."
   INSTRUCTOR_NOTIFICATION_EMAIL = "first@example.edu, second@example.com"
   ```

4. **Save**. Streamlit restarts the app; the keys are picked up on the next run.

## Setting them locally

Either export them in the shell:

```bash
export GOOGLE_API_KEY="..."
streamlit run simulation_app/app.py
```

or write `.streamlit/secrets.toml` (git-ignored) with the same TOML as above.

## Rotating a key

Keys previously committed to this repository remain readable in its git
history. Revoke them at the provider and issue new ones before setting the
secrets above — a key that was ever committed should never be reused.

## Verifying

The admin diagnostics page (`?admin=1` → **LLM** tab) is the place to check:

- **Built-in Provider Keys** lists every slot and whether a key is visible.
- **Test providers now** sends one minimal authenticated request per
  configured slot and reports each as OK (with latency), Failed (with the
  reason), or Not configured. Key material is never displayed, and failure
  text is scrubbed of every configured key before it is shown.

In code, `simulation_app/utils/llm_response_generator.py` exposes
`builtin_provider_key_status()` (which slots have a key),
`missing_builtin_provider_secrets()` (which documented names are still unset)
and `LLMResponseGenerator.verify_providers()` (the sweep behind that button).
The app log line `LLM provider chain: …` lists the resolved chain at startup
and never contains key bytes.
