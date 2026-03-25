# Basketball Modeling Session — User Prompts

## Session 1 (Model Build & First Predictions)

1. Implement the following plan: [full plan document for NCAA Basketball Score Prediction 2026 Tournament — Baio & Blangiardo hierarchical model, numpyro NUTS, 5 major conferences, NegBin2 likelihood]

2. I just dropped a new data file: data/halftime_odds.tsv; the format is different. It has multiple lines for each game_id, and we only need one row per game_id; the relevant columns are home_team, away_team, home_final, away_final; please switch to using this data file.

3. how many games in all

4. please create a graph with defensive strength on one axis and offensive strength on the other; one point per team (labeled)

5. which data file is graph based on?

6. please create a graph of the distribution over point spread for a game between Michigan and Michigan State

7. how did you create this graph; how did you use the model?

8. yes; assume neutral location

9. what NCAA tournament can you model today (with the model you have trained)?

10. run predictions for the first-round matchups only

11. do you need me to tell you what the games are?

12. duke vs siena

13. 3  [chose option 3: flag mid-majors, run all modelable games]

14. see data/firstRound2026.csv

15. for each game, please create graphs of the spread distributions and game total distributions (over under), and drop them in the figs/ directory; label the graphs with enough info to know which teams are being modeled and what is being modeled! also the file names should have this info.

16. please do the same thing for figs/secondRound2026.csv, placing the figures into figs/secondRound/

17. I have added the spread and totals predicted by the betting markets. Please add to your summary table above, the probability according to our model, of i) the favorite beating the market spread and of the market spread being surpassed. Add the market spread and market over to the summary table also for easy viewing.

18. this is great; minor improvement would be to list the 'Model Spread' always for the favorite and always list the favorite first

## Session 2 (After Context Reset)

19. what happened?

20. what was my last request?

21. please just display that last output and your recommendations again.

22. are the model spreads listed for the favorite? it seems like they are not always. Please list the spreads (the plus or minus) for the market favorite

23. give me again the Notable disagreements worth flagging

24. please save the table and the Notable disagreements to roundTwo2026.md

25. also save my sequence of prompts over this entire modeling session to promptsBball.md
