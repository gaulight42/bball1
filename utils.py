"""Plotting and football table functions
"""

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, Whisker
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure, show

__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"


def score_table(df):
    """Function to convert football match dataframe to a table
    Keyword arguments:
    df -- matches dataframe with columns Home, score1, score2, Away
    """
    df = df.assign(
        HomePoints=np.select(
            [df["score1"] > df["score2"], df["score1"] < df["score2"]],
            [3, 0],
            default=1,
        ),
        AwayPoints=np.select(
            [df["score2"] > df["score1"], df["score2"] < df["score1"]],
            [3, 0],
            default=1,
        ),
        HomeGD=df["score1"] - df["score2"],
        AwayGD=df["score2"] - df["score1"],
    )

    home_df = df[["Home", "HomePoints", "HomeGD"]]
    away_df = df[["Away", "AwayPoints", "AwayGD"]]

    home_df = home_df.groupby("Home").sum().rename_axis("Team")
    away_df = away_df.groupby("Away").sum().rename_axis("Team")

    df = pd.merge(home_df, away_df, left_index=True, right_index=True)

    df = df.assign(
        Points=df["HomePoints"] + df["AwayPoints"],
        GD=df["HomeGD"] + df["AwayGD"],
    )

    df = df[["Points", "GD"]].astype(int)

    df = df.sort_values(["Points", "GD"], ascending=[False, False])
    return df


def plot_quality(df):
    """Use bokeh to plot team strengths
    Keyword arguments:
    df -- quality dataframe with columns:
        - Team
        - attack
        - attacksd
        - attack_low
        - attack_high
        - defend
        - defendsd
        - defend_low
        - defend_high
    """
    source = ColumnDataSource(df)
    p = figure(x_range=(-1, 1.5), y_range=(-1, 1.5))
    p.add_layout(
        Whisker(
            lower="attack_low",
            upper="attack_high",
            base="defend",
            dimension="width",
            source=source,
        )
    )
    p.add_layout(
        Whisker(
            lower="defend_low",
            upper="defend_high",
            base="attack",
            dimension="height",
            source=source,
        )
    )
    p.circle(x="attack", y="defend", source=source)

    p.title.text = "Team strengths"
    p.xaxis.axis_label = "Attacking"
    p.yaxis.axis_label = "Defending"

    hover = HoverTool()
    hover.tooltips = [
        ("Team", "@Team"),
        ("Attacking", "@attack"),
        ("Attacking sd", "@attacksd"),
        ("Defending", "@defend"),
        ("Defending sd", "@defendsd"),
    ]
    p.add_tools(hover)

    return p

def plot_quality2(df):
    """Use matplotlib to plot team strengths
    Keyword arguments:
    df -- quality dataframe with columns:
        - Team
        - attack
        - attacksd
        - attack_low
        - attack_high
        - defend
        - defendsd
        - defend_low
        - defend_high
    """
    plt.figure(figsize=(20,10))
    p1 = sns.scatterplot('attack', # Horizontal axis
                        'defend', # Vertical axis
                         data=qDF, # Data source
                         size = 8,
                         legend=False)

    for line in range(0,qDF.shape[0]):
        p1.text(qDF.attack[line]+0.001, qDF.defend[line],
        qDF.Team[line], horizontalalignment='left',
        size='large', color='black', weight='semibold')

    plt.title('Quality')
    plt.xlabel('attack strength')
    plt.ylabel('defend strength')
    return p1
