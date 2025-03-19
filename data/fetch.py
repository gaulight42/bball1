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


year = int(sys.argv[1])
print("fetching %d" % (year))

a = ms.get_games_season(year)

json.dump(json.loads(a[0].to_json()), open(f"season_{year}.info.json","w"))
json.dump(json.loads(a[1].to_json()), open(f"season_{year}.box.json","w"))
json.dump(json.loads(a[2].to_json()), open(f"season_{year}.pbp.json","w"))

