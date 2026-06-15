# Runbook: wiring pyfeat-live releases → py-feat changelog (PAT + dispatch)

**Date:** 2026-06-14
**Status:** Setup steps (manual, one-time) — code side landed in PR #332
**Related:** `.github/workflows/refresh_pyfeatlive_changelog.yml` (PR #332),
`docs/marimo/content/pyfeatlive_changelog.md`, Workstream D of
`2026-06-03-docs-modernization-marimo-book.md`

## What's already automated (no setup needed)

PR #332 adds `refresh_pyfeatlive_changelog.yml` to py-feat. It regenerates
`pyfeatlive_changelog.md` from the **cosanlab/pyfeat-live** GitHub Releases via
`marimo-book sync-releases` and opens a PR when the changelog changes. It fires on:

- **`schedule`** (daily, `17 7 * * *`) — the catch-all; works out of the box.
- **`workflow_dispatch`** — run it manually any time.
- **`repository_dispatch`** (`type: pyfeatlive-release`) — the *instant* path,
  fired by pyfeat-live when it publishes a release. **This is the only part that
  needs setup.**

Because **pyfeat-live is public**, py-feat's refresh job reads its releases with
the default `GITHUB_TOKEN` — **no read-PAT is required on the py-feat side.**

## What you need to set up (one-time)

### 1. Enable auto-PR creation in py-feat

The refresh workflow opens a PR via `peter-evans/create-pull-request`. In
**py-feat → Settings → Actions → General → Workflow permissions**, tick
**"Allow GitHub Actions to create and approve pull requests."** (Without this the
daily/dispatch run fails at the open-PR step.)

### 2. Create a fine-grained PAT for the dispatch (lives in pyfeat-live)

pyfeat-live's release workflow has to POST a `repository_dispatch` to *py-feat*.
Its own `GITHUB_TOKEN` can't reach another repo, so it needs a PAT with write
access to py-feat.

GitHub → **Settings → Developer settings → Fine-grained personal access tokens →
Generate new token**:

- **Resource owner:** `cosanlab`
- **Repository access:** *Only select repositories* → **cosanlab/py-feat**
- **Permissions → Repository permissions → Contents: Read and write**
  (this is what the `POST /repos/.../dispatches` API requires; grant nothing else)
- **Expiration:** set a calendar reminder to rotate it.

Copy the token value (shown once).

> Classic-PAT alternative: a classic token with the `repo` scope also works, but
> it's broader than needed — prefer the fine-grained token above.

### 3. Store the PAT as a secret in pyfeat-live

In **cosanlab/pyfeat-live → Settings → Secrets and variables → Actions → New
repository secret**:

- **Name:** `PYFEAT_DISPATCH_TOKEN`
- **Value:** the PAT from step 2

Or from a checkout of pyfeat-live with `gh` authenticated:

```bash
gh secret set PYFEAT_DISPATCH_TOKEN --repo cosanlab/pyfeat-live
# paste the token at the prompt
```

### 4. Fire the dispatch from pyfeat-live's release workflow

Add a job to pyfeat-live's release workflow (the one that runs on
`release: published`) that pings py-feat:

```yaml
  notify-pyfeat:
    name: Notify py-feat to refresh the changelog
    runs-on: ubuntu-latest
    steps:
      - name: Fire repository_dispatch at py-feat
        uses: peter-evans/repository-dispatch@v3
        with:
          token: ${{ secrets.PYFEAT_DISPATCH_TOKEN }}
          repository: cosanlab/py-feat
          event-type: pyfeatlive-release
```

The `event-type` **must** be `pyfeatlive-release` — it matches the
`repository_dispatch.types` filter in `refresh_pyfeatlive_changelog.yml`.

## Verify end-to-end

1. Publish a (test) release on pyfeat-live, or re-run its release workflow.
2. The `notify-pyfeat` job fires the dispatch.
3. In py-feat → Actions, **"Refresh Py-feat Live changelog"** starts within
   seconds and opens (or updates) the `bot/refresh-pyfeatlive-changelog` PR.
4. Merge that PR; the marimo book picks up the new `pyfeatlive_changelog.md`.

Fallback: if the dispatch isn't wired yet, the daily `schedule` run still catches
new releases within 24h. You can also trigger it manually via the
**Run workflow** button (`workflow_dispatch`).

## Note / open item

PR #332 attached the pyfeat-live section + `release_notes:` block to
**`docs/marimo/book.yml`** (the tutorials scratch book), whereas the canonical
v2.0 site is **`docs/book.yml`**. Before the docs cutover, fold the pyfeat-live
TOC section and the `pyfeatlive_changelog.md` page into `docs/book.yml` so the
generated changelog actually appears on the published site.
