# -*- python -*-

# jhndrsn@acm.org

import json
import sys
import argparse
from datetime import datetime
import cbbpy.mens_scraper as ms

def parse_date(date_str):
    """Convert YYYYMMDD string to datetime object"""
    return datetime.strptime(date_str, '%Y%m%d')

def main():
    parser = argparse.ArgumentParser(description='Fetch a single NCAA basketball game by date and team')
    parser.add_argument('--date', help='Date in YYYYMMDD format (defaults to today)')
    parser.add_argument('--team', required=True, help='Team name (e.g., "Kansas Jayhawks")')
    args = parser.parse_args()

    # Use today's date if none specified
    if args.date is None:
        args.date = datetime.now().strftime('%Y%m%d')

    # Convert date string to datetime
    game_date = parse_date(args.date)
    print(f"Fetching games for {args.team} on {args.date}")

    # Get all game IDs for today
    game_ids = ms.get_game_ids(game_date)
    
    if not game_ids:
        print(f"No games found on {args.date}")
        sys.exit(1)

    # Find the game with our team
    matching_game_id = None
    for game_id in game_ids:
        game_info = ms.get_game_info(game_id)
        if args.team in game_info['home_team'].iloc[0] or args.team in game_info['away_team'].iloc[0]:
            matching_game_id = game_id
            break

    if not matching_game_id:
        print(f"No games found for {args.team} on {args.date}")
        sys.exit(1)

    # Get the full game data
    game_info, game_box, game_pbp = ms.get_game(matching_game_id)

    # Create output filenames with date and team
    base_filename = f"{args.date}_{args.team.replace(' ', '_')}"
    
    # Reset indices and save to JSON in the format expected by torows.py
    json.dump(json.loads(game_info.reset_index().to_json(orient='columns')), open(f"{base_filename}.info.json", "w"))
    json.dump(json.loads(game_box.reset_index().to_json(orient='columns')), open(f"{base_filename}.box.json", "w"))
    json.dump(json.loads(game_pbp.reset_index().to_json(orient='columns')), open(f"{base_filename}.pbp.json", "w"))
    
    print(f"Saved game data to {base_filename}.*.json")

if __name__ == '__main__':
    main()

