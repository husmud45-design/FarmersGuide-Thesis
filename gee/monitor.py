import argparse
import collections
import re
import time

import ee

EE_PROJECT = "farmers-guide-491313"
PREFIX = "fg_modis2_"
QUOTA = 150.0          # Community tier EECU-hours per month

STATE_ORDER = ["SUCCEEDED", "RUNNING", "PENDING", "FAILED", "CANCELLED"]


def collect():
    rows = []
    for t in ee.data.listOperations():
        m = t.get("metadata", {})
        desc = m.get("description", "")
        if not desc.startswith(PREFIX):
            continue
        season = re.match(r"fg_modis_(\d{4}_\d{2})", desc)
        eecu = m.get("batchEecuUsageSeconds")
        rows.append({
            "desc": desc,
            "season": season.group(1) if season else "?",
            "state": m.get("state", "?"),
            "eecu": float(eecu) / 3600 if eecu else 0.0,
            "error": (t.get("error", {}).get("message")
                      or m.get("error_message") or ""),
        })
    return rows


def summarise(rows):
    by_season = collections.defaultdict(collections.Counter)
    eecu_season = collections.Counter()
    for r in rows:
        by_season[r["season"]][r["state"]] += 1
        eecu_season[r["season"]] += r["eecu"]

    states = [s for s in STATE_ORDER
              if any(c[s] for c in by_season.values())]
    hdr = f"{'season':>9s} " + "".join(f"{s[:4]:>7s}" for s in states) + \
          f"{'EECU-h':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for season in sorted(by_season):
        c = by_season[season]
        line = f"{season:>9s} " + "".join(f"{c[s] or '':>7}" for s in states)
        print(line + f"{eecu_season[season]:>10.2f}")
    print("-" * len(hdr))

    tot = collections.Counter()
    for c in by_season.values():
        tot.update(c)
    total_eecu = sum(eecu_season.values())
    print(f"{'TOTAL':>9s} " + "".join(f"{tot[s]:>7d}" for s in states) +
          f"{total_eecu:>10.2f}")

    done = tot["SUCCEEDED"]
    n = sum(tot.values())
    print(f"\n  {done}/{n} succeeded ({100*done/max(n,1):.0f}%)")
    print(f"  quota used: {total_eecu:.1f} of {QUOTA:.0f} EECU-h "
          f"({100*total_eecu/QUOTA:.1f}%)")
    if done and tot["PENDING"] + tot["RUNNING"]:
        rate = total_eecu / done
        left = (tot["PENDING"] + tot["RUNNING"]) * rate
        print(f"  projected to finish: +{left:.1f} EECU-h "
              f"({100*(total_eecu+left)/QUOTA:.0f}% of allowance)")

    complete = [s for s, c in by_season.items()
                if c["SUCCEEDED"] == sum(c.values())]
    if complete:
        print(f"\n  seasons fully complete: {', '.join(sorted(complete))}")


def list_failed(rows):
    bad = [r for r in rows if r["state"] in ("FAILED", "CANCELLED")]
    if not bad:
        print("No failed tasks.")
        return
    print(f"{len(bad)} failed:\n")
    for r in sorted(bad, key=lambda x: x["desc"]):
        print(f"  {r['desc']:42s} {r['error'][:70]}")
    errs = collections.Counter(r["error"][:60] for r in bad)
    print("\n  by cause:")
    for e, n in errs.most_common():
        print(f"    {n:3d}  {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds")
    args = ap.parse_args()

    ee.Initialize(project=EE_PROJECT)
    while True:
        rows = collect()
        if not rows:
            print(f"No tasks matching {PREFIX}*")
            return
        if args.failed:
            list_failed(rows)
        else:
            summarise(rows)
        if not args.watch:
            return
        print(f"\n[refreshing in {args.watch}s — Ctrl-C to stop]\n")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
