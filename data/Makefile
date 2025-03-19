SHELL=bash

# This goes and grabs the data using cbbpy and reformats it into a
# one-line-per-play easy-to-use format. "tweets.jsonl."  Probably not
# easy to use for any one case, but way easier to read and debug while
# you develop your code. 
#
# Change YEAR to the second part of season aka 2024-2025 is 2025.
#
# jhndrsn@acm.org

YEAR=2025

all: top_plays_${YEAR}.txt retrieval season_${YEAR}.tweets.jsonl season_${YEAR}.game_ends.jsonl season_${YEAR}.reg_ends.jsonl season_${YEAR}.half.jsonl

retrieval: season_${YEAR}.pbp.json season_${YEAR}.box.json season_${YEAR}.info.json

top_plays_%.txt: season_%.tweets.jsonl
	jq .play_desc season_$*.tweets.jsonl | sort --parallel=10 | uniq -c | sort -nr | head -n 1000 > $@

season_%.tweets.jsonl: season_%.pbp.jsonl
	cat $< | jq '{game_id,half,secs_left_reg,secs_left_half,home_score,away_score,home_team,away_team,play_desc}' -c > $@

.PRECIOUS: season_%.pbp.jsonl
season_%.pbp.jsonl: season_%.pbp.json
	python torows.py $< > $@

season_%.pbp.json season_%.info.json season_%.box.json:
	python fetch.py $*

season_%.game_ends.jsonl: season_%.tweets.jsonl
	cat $< | jq -c '{game_id,half,secs_left_half,home_score,away_score}' | jq -sc 'sort_by(-.half,.secs_left_half) | .[]' | sort -u -k 2,2 -t: > $@

season_%.reg_ends.jsonl: season_%.tweets.jsonl
	cat $< | jq -c '{game_id,half,secs_left_reg,home_score,away_score}' | jq -sc 'sort_by(.secs_left_reg) | .[]' | sort -u -k 2,2 -t: > $@

season_%.half.jsonl: season_%.tweets.jsonl
	cat $< | jq -c '{game_id,half,secs_left_half,home_score,away_score}' | jq -sc 'sort_by(.half,.secs_left_half) | .[]' | sort -u -k 2,2 -t: > $@
