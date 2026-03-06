# espn_pbp.py - scrape play-by-play data from ESPN
#
# ESPN restructured their JSON in 2025-26, breaking cbbpy's pbp parser.
# This module scrapes pbp directly from ESPN pages using the new format.
#
# jhndrsn@acm.org

import re
import json
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
]

WINDOW_STRING = "window['__espnfitt__']="
JSON_REGEX = r"window\['__espnfitt__'\]={(.*)};"
PBP_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/{}"
ATTEMPTS = 3


def _get_gamepackage(soup):
    for x in soup.find_all("script"):
        if WINDOW_STRING in x.text:
            m = re.search(JSON_REGEX, x.text)
            if m:
                return json.loads("{" + m.group(1) + "}")["page"]["content"]["gamepackage"]
    return None


def get_game_pbp(game_id):
    """Scrape play-by-play for a single game. Returns a DataFrame."""
    game_id = str(game_id)

    for attempt in range(ATTEMPTS):
        try:
            header = {"User-Agent": np.random.choice(USER_AGENTS)}
            page = requests.get(PBP_URL.format(game_id), headers=header)
            soup = bs(page.content, "lxml")
            gp = _get_gamepackage(soup)

            if gp is None:
                if attempt + 1 == ATTEMPTS:
                    print(f"  {game_id}: no gamepackage found")
                    return pd.DataFrame()
                time.sleep(np.random.uniform(1, 3))
                continue

            status = gp.get("gmStrp", {}).get("status", {}).get("desc", "")
            if status not in ("Final", "Final/OT", "Final/2OT", "Final/3OT", "Final/4OT"):
                print(f"  {game_id}: status={status}, skipping")
                return pd.DataFrame()

            return _parse_pbp(gp, game_id)

        except Exception as ex:
            if attempt + 1 == ATTEMPTS:
                print(f"  {game_id}: error: {ex}")
                return pd.DataFrame()
            time.sleep(np.random.uniform(1, 3))

    return pd.DataFrame()


def _parse_pbp(gp, game_id):
    pbp = gp["pbp"]
    tms = pbp.get("tms", {})
    home_team = tms.get("home", {}).get("nm", "")
    away_team = tms.get("away", {}).get("nm", "")

    plays = pbp.get("plays", [])
    if not plays:
        return pd.DataFrame()

    rows = []
    for p in plays:
        home_away = p.get("homeAway", "")
        if home_away == "home":
            team = home_team
        elif home_away == "away":
            team = away_team
        else:
            team = ""

        clock = p.get("clock", {})
        period = p.get("period", {}).get("number", np.nan)
        secs_in_period = clock.get("value", np.nan)

        # compute secs_left_reg: men's basketball uses 20-min halves
        if not np.isnan(period) and not np.isnan(secs_in_period):
            if period == 1:
                secs_left_reg = 1200 + secs_in_period
            elif period == 2:
                secs_left_reg = secs_in_period
            else:
                secs_left_reg = 0  # OT
        else:
            secs_left_reg = np.nan

        athlete = p.get("athlete", {})

        rows.append({
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "text": p.get("text", ""),
            "home_score": p.get("hmScr", np.nan),
            "away_score": p.get("awScr", np.nan),
            "half": period,
            "secs_left_half": secs_in_period,
            "secs_left_reg": secs_left_reg,
            "play_team": team,
            "play_type": p.get("type", {}).get("txt", ""),
            "shooting_play": p.get("shootingPlay", False),
            "scoring_play": p.get("scoringPlay", False),
            "player": athlete.get("name", ""),
            "player_id": athlete.get("id", ""),
            "coordinate_x": p.get("coordinate", {}).get("x", np.nan),
            "coordinate_y": p.get("coordinate", {}).get("y", np.nan),
        })

    return pd.DataFrame(rows)
