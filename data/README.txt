This directory contains the code for fetching data for a season of
college basketball. Debugged and runs on ubuntu 22.04

Create a conda using the environment.yml first

```bash
conda env create -f environment.yml
```

then just "make" to run the default target. It will fetch an entire season.

Takes about 15 minutes to fetch and reformat all the data on a gigabit
internet connection.


TO GET ONE GAME:

```bash
make get_one_game DATE=20250319 TEAM=Spartans
```

will produce 20250319_Spartans.tweets.jsonl

Should only take a few seconds to get one game.

---

jhndrsn@acm.org
20250319
