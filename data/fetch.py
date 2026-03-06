# -*- python -*-

# Just scrape the season! We'll convert it after. Don't try to do too
# much at once, or we'll have to refetch the entire thing (boo).
# Takes a little while because it iterates through days to find days
# with games. And then loads games one at a time. But it's automated
# and the authors of cbbpy did a great job!
#
# jhndrsn@acm.org

import json
import sys
import joblib

# Force joblib to use threads so our monkey-patch is visible to all workers.
# cbbpy uses joblib.Parallel with loky (multiprocessing) by default, which
# spawns new processes that don't inherit monkey-patches. Threading is fine
# here since the work is I/O-bound (web scraping).
joblib.parallel.DEFAULT_BACKEND = 'threading'

import cbbpy.utils.cbbpy_utils as _cbbpy_utils
import cbbpy.mens_scraper as ms

# Monkey-patch: ESPN removed 'isConferenceGame' from their API JSON in 2025-26.
# cbbpy 2.1.2 expects it at gamepackage["gmStrp"]["isConferenceGame"].
_orig_get_game_info_helper = _cbbpy_utils._get_game_info_helper
def _patched_get_game_info_helper(gamepackage, game_id, game_type):
    gamepackage.setdefault("gmStrp", {}).setdefault("isConferenceGame", False)
    return _orig_get_game_info_helper(gamepackage, game_id, game_type)
_cbbpy_utils._get_game_info_helper = _patched_get_game_info_helper

year = int(sys.argv[1])
print("fetching %d" % (year))

# Use cbbpy for info + box (pbp=False to skip its broken pbp parser).
a = ms.get_games_season(year, info=True, box=True, pbp=False)

json.dump(json.loads(a[0].to_json()), open(f"season_{year}.info.json","w"))
json.dump(json.loads(a[1].to_json()), open(f"season_{year}.box.json","w"))

# Scrape pbp ourselves since ESPN restructured their JSON and broke cbbpy.
# Get game IDs from the info we just scraped.
import pandas as pd
from espn_pbp import get_game_pbp
from joblib import Parallel, delayed

info_df = a[0]
game_ids = info_df["game_id"].unique().tolist()
print(f"scraping pbp for {len(game_ids)} games")

from tqdm import tqdm
pbp_dfs = Parallel(n_jobs=8, prefer="threads")(
    delayed(get_game_pbp)(gid) for gid in tqdm(game_ids, desc="Scraping pbp")
)
pbp_df = pd.concat([df for df in pbp_dfs if len(df) > 0], ignore_index=True)
json.dump(json.loads(pbp_df.to_json()), open(f"season_{year}.pbp.json","w"))
print(f"done: {len(pbp_df)} plays across {pbp_df['game_id'].nunique()} games")
