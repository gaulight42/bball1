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
import cbbpy.mens_scraper as ms
import cbbpy.utils.cbbpy_utils as _cbbpy_utils

# Monkey-patch: ESPN removed 'isConferenceGame' from their API JSON in 2025-26.
# cbbpy 2.1.2 expects it at gamepackage["gmStrp"]["isConferenceGame"].
_orig_get_game_info_helper = _cbbpy_utils._get_game_info_helper
def _patched_get_game_info_helper(gamepackage, game_id, game_type):
    gamepackage.setdefault("gmStrp", {}).setdefault("isConferenceGame", False)
    return _orig_get_game_info_helper(gamepackage, game_id, game_type)
_cbbpy_utils._get_game_info_helper = _patched_get_game_info_helper


year = int(sys.argv[1])
print("fetching %d" % (year))

a = ms.get_games_season(year)

json.dump(json.loads(a[0].to_json()), open(f"season_{year}.info.json","w"))
json.dump(json.loads(a[1].to_json()), open(f"season_{year}.box.json","w"))
json.dump(json.loads(a[2].to_json()), open(f"season_{year}.pbp.json","w"))

