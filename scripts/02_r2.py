"""R2: audit published v2 by default; opt in to candidate processing."""

import argparse

from r2_preprocessing import ROOT, run
from r2_validation import audit_release, make_epoch_qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", action="store_true", help="process raw into _candidate only")
    parser.add_argument("--download", action="store_true", help="allow oddball raw download with --process")
    parser.add_argument("-s", "--subjects", nargs="+", help="configured two-digit subject IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tests", action="store_true", help="run synthetic integration checks")
    parser.add_argument("--review-bad", action="append", default=[], metavar="SUBJECT:CHANNEL")
    args = parser.parse_args()
    if (args.download or args.overwrite or args.subjects or args.review_bad) and not args.process:
        parser.error("--download, --overwrite and --subjects require --process")
    if args.tests:
        from r2_checks import test_r2

        test_r2()
    review_bads = {}
    for value in args.review_bad:
        if ":" not in value:
            parser.error("--review-bad requires SUBJECT:CHANNEL")
        subject, channel = value.split(":", 1)
        if not channel:
            parser.error("Empty manual channel name")
        review_bads.setdefault(subject, []).append(channel)
    if args.process:
        run(
            ROOT,
            subjects=args.subjects,
            overwrite=args.overwrite,
            download=args.download,
            review_bads=review_bads or None,
        )
    else:
        audit_release(ROOT)
        make_epoch_qc(ROOT)


if __name__ == "__main__":
    main()
