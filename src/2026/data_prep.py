"""
data_prep.py — Parse, filter, deduplicate, and split 2025-26 NCAA basketball data.

Source: data/halftime_odds.tsv
  - Multiple rows per game_id (one per bookmaker) → deduplicate on game_id
  - Relevant columns: home_team, away_team, home_final, away_final, is_neutral
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Conference rosters (2025-26 season, full "Team Mascot" names
# as they appear in halftime_odds.tsv)
# ──────────────────────────────────────────────────────────────
CONFERENCE_TEAMS = {
    "ACC": [
        "Boston College Eagles", "Clemson Tigers", "Duke Blue Devils",
        "Florida State Seminoles", "Georgia Tech Yellow Jackets",
        "Louisville Cardinals", "Miami Hurricanes", "NC State Wolfpack",
        "Notre Dame Fighting Irish", "Pittsburgh Panthers", "Syracuse Orange",
        "Virginia Cavaliers", "Virginia Tech Hokies", "Wake Forest Demon Deacons",
        "North Carolina Tar Heels",
        # 2024-25 additions
        "California Golden Bears", "Stanford Cardinal", "SMU Mustangs",
    ],
    "Big Ten": [
        "Illinois Fighting Illini", "Indiana Hoosiers", "Iowa Hawkeyes",
        "Maryland Terrapins", "Michigan Wolverines", "Michigan State Spartans",
        "Minnesota Golden Gophers", "Nebraska Cornhuskers", "Northwestern Wildcats",
        "Ohio State Buckeyes", "Penn State Nittany Lions", "Purdue Boilermakers",
        "Rutgers Scarlet Knights", "Wisconsin Badgers",
        # 2024-25 additions
        "UCLA Bruins", "USC Trojans", "Oregon Ducks", "Washington Huskies",
    ],
    "Big 12": [
        "Baylor Bears", "Iowa State Cyclones", "Kansas Jayhawks",
        "Kansas State Wildcats", "Oklahoma State Cowboys", "TCU Horned Frogs",
        "Texas Tech Red Raiders", "West Virginia Mountaineers",
        # Former Pac-12 additions
        "Arizona Wildcats", "Arizona State Sun Devils", "BYU Cougars",
        "Colorado Buffaloes", "Utah Utes",
        # Other additions
        "Cincinnati Bearcats", "Houston Cougars", "UCF Knights",
    ],
    "SEC": [
        "Alabama Crimson Tide", "Arkansas Razorbacks", "Auburn Tigers",
        "Florida Gators", "Georgia Bulldogs", "Kentucky Wildcats",
        "LSU Tigers", "Mississippi State Bulldogs", "Missouri Tigers",
        "Ole Miss Rebels", "South Carolina Gamecocks", "Tennessee Volunteers",
        "Texas A&M Aggies", "Vanderbilt Commodores",
        # 2024-25 additions
        "Oklahoma Sooners", "Texas Longhorns",
    ],
    "Big East": [
        "Butler Bulldogs", "UConn Huskies", "Creighton Bluejays",
        "DePaul Blue Demons", "Georgetown Hoyas", "Marquette Golden Eagles",
        "Providence Friars", "St. John's Red Storm", "Seton Hall Pirates",
        "Villanova Wildcats", "Xavier Musketeers",
    ],
}

DEFAULT_CONFERENCES = list(CONFERENCE_TEAMS.keys())


def load_raw_tsv(path: str | Path) -> pd.DataFrame:
    """
    Load halftime_odds.tsv.

    Deduplicates on game_id (keeping one row per game — scores are identical
    across bookmaker rows), and returns only the columns needed for the model.
    """
    df = pd.read_csv(path, sep="\t")

    # All bookmaker rows for the same game have identical score/team/neutral columns;
    # keep the first occurrence per game_id.
    df = df.drop_duplicates(subset="game_id", keep="first")

    # Rename to canonical names used throughout the pipeline
    df = df.rename(columns={
        "home_final": "home_score",
        "away_final": "away_score",
    })

    # is_neutral arrives as a string "True"/"False" or bool
    df["is_neutral"] = df["is_neutral"].map(
        lambda v: 1 if str(v).strip().lower() == "true" else 0
    )

    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df[["game_id", "home_team", "away_team",
               "home_score", "away_score", "is_neutral"]].reset_index(drop=True)


# Keep load_raw as an alias so the notebook import works unchanged
load_raw = load_raw_tsv


def filter_to_conferences(
    df: pd.DataFrame,
    conferences: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep only games where BOTH teams belong to the target conferences.
    """
    if conferences is None:
        conferences = DEFAULT_CONFERENCES

    team_to_conf: dict[str, str] = {}
    for conf, teams in CONFERENCE_TEAMS.items():
        if conf in conferences:
            for t in teams:
                team_to_conf[t] = conf

    conf_teams = set(team_to_conf.keys())
    mask = df["home_team"].isin(conf_teams) & df["away_team"].isin(conf_teams)
    out = df[mask].copy()
    out["home_conf"] = out["home_team"].map(team_to_conf)
    out["away_conf"] = out["away_team"].map(team_to_conf)
    return out


def build_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean one-row-per-game table.

    The TSV is already deduplicated by load_raw_tsv; this step just ensures
    the output schema matches what encode_teams / inference expect.
    """
    cols = ["home_team", "away_team", "home_score", "away_score", "is_neutral"]
    return df[cols].reset_index(drop=True)


def encode_conferences(
    team_index: pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    """
    Build a conference index aligned with team_index.

    Returns
    -------
    conf_index     : Series mapping conference name → integer id (alphabetical)
    team_conf_id   : int array shape (n_teams,) — conference id for each team,
                     indexed by team_id (i.e. team_conf_id[team_id] = conf_id)
    """
    # Build team → conference lookup
    team_to_conf: dict[str, str] = {}
    for conf, teams in CONFERENCE_TEAMS.items():
        for t in teams:
            team_to_conf[t] = conf

    conf_names = sorted(set(team_to_conf.values()))
    conf_index = pd.Series(
        {name: i for i, name in enumerate(conf_names)}, name="conf_id"
    )

    team_conf_id = np.array([
        conf_index[team_to_conf[team_name]]
        for team_name in team_index.index
    ], dtype=np.int32)

    return conf_index, team_conf_id


def encode_teams(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assign integer IDs to teams (alphabetically sorted).

    Returns
    -------
    games_with_ids : DataFrame with 'home_id' and 'away_id' columns added.
    team_index     : Series mapping team_name → integer id.
    """
    all_teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    team_index = pd.Series(
        {name: i for i, name in enumerate(all_teams)}, name="team_id"
    )
    games = games.copy()
    games["home_id"] = games["home_team"].map(team_index)
    games["away_id"] = games["away_team"].map(team_index)
    return games, team_index


def train_holdout_split(
    games: pd.DataFrame,
    holdout_n: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train / holdout, guaranteeing every holdout team also appears
    in training (so posterior samples exist for all holdout teams).
    """
    rng = np.random.default_rng(seed)

    for _ in range(1000):
        candidate_idx = rng.choice(len(games), size=holdout_n, replace=False)
        holdout = games.iloc[candidate_idx]
        train = games.drop(games.index[candidate_idx])

        train_teams = set(train["home_team"]) | set(train["away_team"])
        holdout_teams = set(holdout["home_team"]) | set(holdout["away_team"])

        if holdout_teams <= train_teams:
            return train.reset_index(drop=True), holdout.reset_index(drop=True)

    raise RuntimeError(
        "Could not find a valid holdout split after 1000 tries. "
        "Try reducing holdout_n."
    )


def prepare_data(
    tsv_path: str | Path,
    conferences: list[str] | None = None,
    holdout_n: int = 30,
    seed: int = 42,
) -> dict:
    """
    Full pipeline: load TSV → filter → encode teams → encode conferences → split.

    Returns a dict with keys:
        train, holdout, team_index, n_teams, all_games,
        conf_index, team_conf_id, n_confs
    """
    raw = load_raw_tsv(tsv_path)
    filtered = filter_to_conferences(raw, conferences)
    games = build_game_table(filtered)
    games, team_index = encode_teams(games)
    conf_index, team_conf_id = encode_conferences(team_index)
    train, holdout = train_holdout_split(games, holdout_n=holdout_n, seed=seed)

    return {
        "train": train,
        "holdout": holdout,
        "team_index": team_index,
        "n_teams": len(team_index),
        "all_games": games,
        "conf_index": conf_index,
        "team_conf_id": team_conf_id,
        "n_confs": len(conf_index),
    }
