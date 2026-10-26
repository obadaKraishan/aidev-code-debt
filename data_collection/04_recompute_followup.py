"""
============================================================================
 04_recompute_followup.py, Recompute Post-Merge Measures From Git History
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : v1.0.0 counted each PR's own squash or merge commit as a
           "post-merge" commit (02_enrich_github.py compared timestamps as
           text: "...Z" sorts after "...+00:00" at the same second) and
           flagged a revert on any message containing the word "revert".
           This script rebuilds the RQ3 measures from git history, with no
           GitHub API calls, for every merged PR in the maintenance
           population:
             1. shallow, partial clone of each repository's default branch
                (commits and trees since 2024-11-01; blobs on demand);
             2. find the commit(s) that landed the PR (committer time equal
                to merged_at, or a message naming the PR; then side branches;
                then the PR head ref), and take the most-changed file of
                that change;
             3. commits touching that file from merged_at to +90 days, first
                100 (the REST endpoint returned one page of 100);
             4. ORIGINAL measures (author date >= merge second, any
                "revert" word), reproduced to validate the method, and
                CORRECTED ones: strictly after merge, excluding the PR's own
                commits (its landing commits and any non-revert commit that
                names the PR), and three revert definitions (any "revert"
                word; a revert commit, i.e. a subject starting "Revert" or
                git's "This reverts commit"; a revert of this PR).
 Input   : data/processed/master.csv
 Output  : data/interim/followup_recomputed.csv (one row per merged PR)
           data/interim/followup_recompute/ (resumable progress files)
 Usage   : python data_collection/04_recompute_followup.py [--workers 8]
============================================================================
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

STATE = config.DATA_INTERIM / "followup_recompute"
WORK = str(STATE / "work")
RES = str(STATE / "results.jsonl")
DONE = str(STATE / "done_repos.txt")
OUT_CSV = config.DATA_INTERIM / "followup_recomputed.csv"
FOLLOWUP_DAYS = 90

ANY_REVERT = re.compile(r"\brevert(s|ed|ing)?\b", re.IGNORECASE)
REVERT_SUBJECT = re.compile(r"^\s*Revert\b", re.IGNORECASE)

def is_revert(msg):
    """A revert commit: the subject line starts with "Revert", or git's "This reverts commit" appears."""
    subject = msg.strip().split("\n", 1)[0] if msg else ""
    return bool(REVERT_SUBJECT.match(subject)) or "This reverts commit" in (msg or "")
SINCE = '2024-11-01'

def git(args, cwd, timeout=900, check=True):
    r = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, timeout=timeout,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace')[:400])
    return r.stdout.decode('utf-8', 'replace')

def clone(repo, n_prs=0):
    d = os.path.join(WORK, hashlib.sha1(repo.encode()).hexdigest()[:16])
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    # repositories with many PRs get their blobs up front, so hundreds of diffs need no network round trips
    blob = [] if n_prs > 300 else ['--filter=blob:none']
    r = subprocess.run(['git', 'clone', '-q'] + blob + ['--no-checkout', '--single-branch',
                        f'--shallow-since={SINCE}', f'https://github.com/{repo}.git', d],
                       capture_output=True, timeout=3600, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if r.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        return None, r.stderr.decode('utf-8', 'replace')[-300:]
    return d, None

def load_log(d, first_parent=True):
    """Commits since SINCE (first-parent chain, or everything reachable): sha, committer epoch, parents, message."""
    raw = git(['log'] + (['--first-parent'] if first_parent else []) + ['--format=%H%x1f%ct%x1f%P%x1f%B%x1e', 'HEAD'], d)
    out = []
    for rec in raw.split('\x1e'):
        rec = rec.strip('\n')
        if not rec:
            continue
        sha, ct, parents, msg = rec.split('\x1f', 3)
        out.append((sha, int(ct), parents.split(), msg))
    return out  # newest first

def numstat(d, a, b):
    raw = git(['diff', '--numstat', '-z', '-M', a, b], d, timeout=1800)
    parts = raw.split('\0')
    files, i = [], 0
    while i < len(parts):
        p = parts[i]
        if not p:
            i += 1; continue
        f = p.split('\t')
        if len(f) >= 3 and f[2] != '':
            path = f[2]; i += 1
        else:  # rename: path follows as two NUL-separated fields
            path = parts[i + 2] if i + 2 < len(parts) else ''; i += 3
        add = 0 if f[0] == '-' else int(f[0]); dele = 0 if f[1] == '-' else int(f[1])
        files.append((path, add + dele))
    return files

def landing(log, merged_epoch, number, log_all):
    """Commits that landed the PR. First the first-parent chain (same committer second +/-2 s, preferring a
    message naming the PR); then any commit reachable from HEAD (PRs merged into a side branch that later
    reached the default branch). Returns (diff base, diff tip, method, set of the PR's own commit shas)."""
    tag = re.compile(rf'(#{number}\b|pull request #{number}\b)')
    near = [i for i, c in enumerate(log) if abs(c[1] - merged_epoch) <= 2]
    if near:
        tagged = [i for i in near if tag.search(log[i][3])]
        idx = tagged if tagged else near
        method = 'time+tag' if tagged else 'time'
    else:
        idx = [i for i, c in enumerate(log) if tag.search(c[3]) and abs(c[1] - merged_epoch) <= 86400]
        method = 'tag'
        idx = idx[-1:]
    if idx:
        newest, oldest = min(idx), max(idx)
        if any(i not in idx for i in range(newest, oldest + 1)):
            newest = oldest = idx[0]
        parents = log[oldest][2]
        if parents:
            own = {log[i][0] for i in range(newest, oldest + 1)}
            return parents[0], log[newest][0], method, own
    # side-branch fallback over all commits
    cands = [c for c in log_all if tag.search(c[3]) and abs(c[1] - merged_epoch) <= 86400]
    if not cands:
        cands = [c for c in log_all if abs(c[1] - merged_epoch) <= 2]
        if len(cands) != 1:
            return None, None, 'unresolved', set()
    c = min(cands, key=lambda c: abs(c[1] - merged_epoch))
    if not c[2]:
        return None, None, 'no-parent', set()
    return c[2][0], c[0], 'side-branch', {c[0]}

def history(d, path, since_iso, until_iso):
    raw = git(['log', '-n', '100', f'--since={since_iso}', f'--until={until_iso}',
               '--format=%H%x1f%aI%x1f%B%x1e', 'HEAD', '--', path], d)
    out = []
    for rec in raw.split('\x1e'):
        rec = rec.strip('\n')
        if not rec:
            continue
        sha, ai, msg = rec.split('\x1f', 2)
        out.append((sha, pd.Timestamp(ai).tz_convert('UTC'), msg))
    return out

def bulk_index(d):
    """One pass over every reachable commit: file -> [(committer epoch, sha, author date UTC, message)], newest first."""
    raw = git(['log', '--cc', '--no-renames', '--name-only', '--format=%x1e%H%x1f%aI%x1f%ct%x1f%B%x1d', 'HEAD'], d, timeout=3600)
    idx = {}
    for rec in raw.split('\x1e'):
        if not rec.strip():
            continue
        head, _, files = rec.partition('\x1d')
        sha, ai, ct, msg = head.split('\x1f', 3)
        item = (int(ct), sha, pd.Timestamp(ai).tz_convert('UTC'), msg.strip('\n'))
        for f in files.strip('\n').split('\n'):
            if f:
                idx.setdefault(f, []).append(item)
    for f in idx:
        idx[f].sort(key=lambda t: -t[0])
    return idx

def history_bulk(idx, path, since_epoch, until_epoch):
    return [(sha, ai, msg) for ct, sha, ai, msg in idx.get(path, []) if since_epoch <= ct <= until_epoch][:100]

def prefetch_blobs(d, pairs):
    """Fetch, in one round trip, every blob the numstat diffs of these (base, tip) pairs will need."""
    oids = set()
    for a, b in pairs:
        try:
            raw = git(['diff', '--raw', '--no-renames', '-z', '--abbrev=40', a, b], d, timeout=600)
        except Exception:
            continue
        for rec in raw.split(':')[1:]:
            f = rec.split(' ')
            if len(f) >= 4:
                for oid in (f[2], f[3]):
                    if oid and set(oid) != {'0'}:
                        oids.add(oid)
    if not oids:
        return
    need = sorted(oids)
    for i in range(0, len(need), 2000):
        subprocess.run(['git', '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '-q', 'origin', '--no-tags',
                        '--no-write-fetch-head', '--recurse-submodules=no', '--filter=blob:none', '--stdin'],
                       cwd=d, input='\n'.join(need[i:i + 2000]).encode(), capture_output=True, timeout=1800,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})

def process(repo, prs):
    d, err = clone(repo, len(prs))
    rows = []
    if d is None:
        for _, x in prs.iterrows():
            rows.append({'id': int(x.id), 'repo': repo, 'status': 'clone_failed', 'err': err})
        return rows
    try:
        log = load_log(d)
        log_all = load_log(d, first_parent=False)
        idx = bulk_index(d) if len(prs) > 150 else None
        checked, bulk_ok = 0, idx is not None
        # pass 1: where did each PR land?
        plan = []
        for _, x in prs.iterrows():
            md = pd.Timestamp(x.merged_at)
            md = md.tz_convert('UTC') if md.tzinfo else md.tz_localize('UTC')
            a, b, method, own = landing(log, int(md.timestamp()), int(x.number), log_all)
            plan.append([x, md, a, b, method, own])
        # PRs that never reached the default branch: fetch all their head refs at once
        lost = [p_ for p_ in plan if p_[2] is None]
        if lost:
            specs = [f'+refs/pull/{int(p_[0].number)}/head:refs/remotes/pr/{int(p_[0].number)}' for p_ in lost]
            for i in range(0, len(specs), 200):
                subprocess.run(['git', 'fetch', '-q', f'--shallow-since={SINCE}', 'origin'] + specs[i:i + 200], cwd=d,
                               capture_output=True, timeout=1800, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
            for p_ in lost:
                ref = f'refs/remotes/pr/{int(p_[0].number)}'
                try:
                    base = git(['merge-base', 'HEAD', ref], d).strip()
                except Exception:
                    base = ''
                if base:
                    p_[2], p_[3], p_[4] = base, ref, 'pr-head'
        # pass 2: one round trip for all blobs the diffs need (partial clones only)
        if len(prs) <= 300:
            prefetch_blobs(d, [(p_[2], p_[3]) for p_ in plan if p_[2]])
        for x, md, a, b, method, own in plan:
            row = {'id': int(x.id), 'repo': repo, 'number': int(x.number), 'group': x.author_group, 'landing': method}
            try:
                if not a:
                    row['status'] = method; rows.append(row); continue
                files = numstat(d, a, b)
                if not files:
                    row['status'] = 'empty_diff'; rows.append(row); continue
                top = max(files, key=lambda f: f[1])[0]
                until = md + timedelta(days=FOLLOWUP_DAYS)
                if bulk_ok:
                    hist = history_bulk(idx, top, int(md.timestamp()), int(until.timestamp()))
                    if checked < 25:  # verify the shortcut on this repository before trusting it
                        ref = history(d, top, md.isoformat(), until.isoformat())
                        checked += 1
                        if {h[0] for h in ref} != {h[0] for h in hist}:
                            bulk_ok = False
                            hist = ref
                else:
                    hist = history(d, top, md.isoformat(), until.isoformat())
                row['bulk'] = int(bulk_ok)
                tag = re.compile(rf'#{int(x.number)}\b')
                title = str(x.title).strip().lower() if isinstance(x.title, str) else ''
                old = [h for h in hist if h[1] >= md.floor('s')]
                after = [h for h in hist if h[1] > md and h[0] not in own]
                # the PR's own commits: its landing commits, and any non-revert commit that names the PR
                new = [h for h in after if is_revert(h[2]) or not tag.search(h[2])]
                row.update({
                    'status': 'ok', 'top_file': top, 'n_hist': len(hist),
                    'old_count': len(old), 'old_revert': int(any(ANY_REVERT.search(h[2]) for h in old)),
                    'new_count': len(new),
                    'revert_any_word': int(any(ANY_REVERT.search(h[2]) for h in new)),
                    'revert_format': int(any(is_revert(h[2]) for h in new)),
                    'revert_of_this_pr': int(any(is_revert(h[2]) and (tag.search(h[2]) or (len(title) > 12 and title[:60] in h[2].lower())) for h in new)),
                    'own_in_old': int(any(h[0] in own or (tag.search(h[2]) and not is_revert(h[2])) for h in old)),
                    'hist': [[h[0][:12], int(h[1].timestamp()), h[2].strip().split('\n', 1)[0][:100], int(is_revert(h[2])),
                              int(bool(tag.search(h[2]))), int(h[0] in own)] for h in hist],
                })
            except Exception as e:  # keep going on any single-PR failure
                row['status'] = 'error'; row['err'] = str(e)[:300]
            rows.append(row)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rows

COLUMNS = {"group": "author_group", "new_count": "followup_commits_90d",
           "old_count": "followup_commits_90d_original", "old_revert": "revert_original"}


def finalize():
    rows = [json.loads(l) for l in open(RES)]
    df = pd.DataFrame(rows).rename(columns=COLUMNS)
    df = df.drop_duplicates("id", keep="last")
    keep = ["id", "repo", "number", "author_group", "status", "landing", "top_file",
            "followup_commits_90d", "followup_commits_90d_original", "revert_original",
            "revert_any_word", "revert_format", "revert_of_this_pr", "own_in_old"]
    df[[c for c in keep if c in df.columns]].to_csv(OUT_CSV, index=False)
    ok = df[df.status == "ok"]
    print(f"  [SAVED] {OUT_CSV}  ({len(ok):,} of {len(df):,} PRs recomputed)")
    print(df.status.value_counts().to_string())


def main(workers=8):
    os.makedirs(WORK, exist_ok=True)
    m = pd.read_csv(config.FILE_MASTER, low_memory=False)
    m = m[(m.is_merged == 1) & m.post_merge_commits_90d.notna()]
    done = set(open(DONE).read().split("\n")) if os.path.exists(DONE) else set()
    groups = m.groupby("repo")
    repos = sorted((r for r in groups.groups if r not in done), key=lambda r: -len(groups.get_group(r)))
    print(f"  {len(m):,} merged PRs in {m.repo.nunique():,} repositories; {len(repos):,} still to process")
    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(RES, "a") as fo, open(DONE, "a") as fd:
        futs = {ex.submit(process, r, groups.get_group(r)): r for r in repos}
        for f in as_completed(futs):
            r = futs[f]
            try:
                rows = f.result()
            except Exception as e:
                rows = [{"repo": r, "status": "repo_error", "err": str(e)[:300]}]
            for row in rows:
                fo.write(json.dumps(row) + "\n")
            fo.flush(); fd.write(r + "\n"); fd.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n:,}/{len(repos):,} repositories  {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    finalize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
