"""
Analyze Team Statistics
=======================
"""

import pandas as pd

from training_v2.config import TEAM_DATASET_PATH


def load_dataset():

    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)

    print(df.shape)

    return df

def analyze_team_statistics(df):

    print("\n" + "=" * 60)
    print("TEAM STATISTICS")
    print("=" * 60)

    summary = (

        df.groupby("Team")

        .agg(

            Tournament=("Tournament", "count"),

            Years=("Year", "nunique"),

            Maps=("Total Maps Played", "sum"),

            Average_Maps=("Total Maps Played", "mean"),

        )

        .sort_values(

            by="Maps",

            ascending=False,

        )

    )

    print(summary)

    return summary

def main():

    df = load_dataset()

    analyze_team_statistics(df)


if __name__ == "__main__":
    main()