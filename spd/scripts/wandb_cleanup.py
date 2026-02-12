"""Bulk-delete old WandB runs to free up storage."""

import argparse
from datetime import UTC, datetime, timedelta
from typing import Any

import wandb


def get_matching_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    cutoff: datetime | None,
    output_dir_substr: str | None,
) -> list[Any]:
    runs = api.runs(f"{entity}/{project}")
    matched = []
    for run in runs:
        if cutoff is not None:
            created = datetime.fromisoformat(run.created_at).replace(tzinfo=UTC)
            if created >= cutoff:
                continue
        if output_dir_substr is not None:
            run_output_dir = run.config.get("output_dir_name") or ""
            if output_dir_substr not in run_output_dir:
                continue
        matched.append(run)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete old WandB runs to free storage.")
    parser.add_argument("--entity", required=True, help="WandB entity (username or team)")
    parser.add_argument("--project", help="Specific project (if omitted, iterate all projects)")
    parser.add_argument("--days", type=int, help="Only delete runs older than this many days")
    parser.add_argument(
        "--output-dir-name", help="Only delete runs whose output_dir_name contains this substring"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Actually delete runs (default is dry run)"
    )
    args = parser.parse_args()

    browse_mode = args.days is None and args.output_dir_name is None

    api = wandb.Api()
    cutoff = datetime.now(tz=UTC) - timedelta(days=args.days) if args.days is not None else None

    if args.project:
        projects = [args.project]
    else:
        projects = [p.name for p in api.projects(entity=args.entity)]

    filters: list[str] = []
    if args.days is not None:
        filters.append(f"older than {args.days} days")
    if args.output_dir_name is not None:
        filters.append(f"output_dir_name contains '{args.output_dir_name}'")
    filter_desc = " AND ".join(filters)

    if browse_mode:
        print("No filters specified, listing all runs (delete disabled).")

    total_matched = 0
    for project in projects:
        matched = get_matching_runs(api, args.entity, project, cutoff, args.output_dir_name)
        if not matched:
            continue

        if browse_mode:
            print(f"\n{project}: {len(matched)} runs")
        else:
            action = "Deleting" if args.delete else "Would delete"
            print(f"\n{project}: {action} {len(matched)} runs ({filter_desc})")

        for run in matched:
            output_dir = run.config.get("output_dir_name", "N/A")
            print(f"  {run.id} | {run.name} | {output_dir} | created {run.created_at}")
            if args.delete and not browse_mode:
                run.delete()

        total_matched += len(matched)

    if browse_mode:
        print(f"\nFound {total_matched} runs across {len(projects)} projects")
    else:
        action = "Deleted" if args.delete else "Would delete"
        print(f"\n{action} {total_matched} runs across {len(projects)} projects")
        if not args.delete and total_matched > 0:
            print("Run with --delete to actually delete these runs.")


if __name__ == "__main__":
    main()
