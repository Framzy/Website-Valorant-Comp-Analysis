import pandas as pd

from training.team.preprocessing.build_team_dataset import (
    load_dataset,
    aggregate_matches,
    split_dataset,
    resolve_composition_records,
)


df = load_dataset()

aggregated = aggregate_matches(df)

exact_five, greater_than_five = split_dataset(aggregated)

results = []

for _, row in greater_than_five.iterrows():

    records = resolve_composition_records(
        raw_df=df,
        tournament=row["Tournament"],
        team=row["Team"],
        map_name=row["Map"],
    )

    for record in records:
        if record.get("__resolved_from") == "role_based_guess":
            results.append(record)


print("\n" + "=" * 80)
print("ROLE-BASED GUESS AUDIT")
print("=" * 80)

print(f"Residual records : {len(results)}")

if not results:
    print("No role-based guess records found.")
    raise SystemExit


audit = pd.DataFrame(results)

columns = [
    "Tournament",
    "Stage",
    "Match Type",
    "Team",
    "Map",
    "Year",
    "Agent",
    "Total Maps Played",
    "Total Wins By Map",
    "Total Loss By Map",
]

print("\n--- 16 RESIDUAL RECORDS ---\n")
print(
    audit[columns]
    .sort_values(["Year", "Team", "Map"])
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Residual records : {len(audit)}")
print(
    f"Total maps affected : "
    f"{audit['Total Maps Played'].sum()}"
)

print(
    f"Total wins affected : "
    f"{audit['Total Wins By Map'].sum()}"
)

print(
    f"Total losses affected : "
    f"{audit['Total Loss By Map'].sum()}"
)