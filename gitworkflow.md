# Git Workflow

- Below I describe my personal git workflows. There are of course many ways of doing this but this is what works for me. Since I'm just describing my workflow, there will be some repetition of commands.
- I don't consider myself a git wizard (aka a gizard), so I generally avoid using a lot of extra flags. They probably do a lot of cool things, but to make my workflow simple and easy to remember, I don't use many of them.

### Development Workflow

0. Currently on `main` branch.
1. Verify that local `main` has the latest changes from the remote before beginning work on the feature --> `git pull`.
2. Now create the feature branch off of `main` --> `git checkout -b my_feat_branch`. This creates a new local branch with name `my_feat_branch`, and automatically checks it out.
3. Do some work on `my_feat_branch`.
4. Stage the changes for committing -> `git add <relative_file_path_of_changes>`. E.g. setting `relative_file_path_of_changes` to `.` will add all files in the working directory recursively. You can view the staged and unstaged changes on the branch using `git status`.
5. Commit the staged changes `git commit -m "my commit message"`.
6. Repeat steps 3, 4, and 5 until the branch is ready to be reviewed.
   - Note, you'll want to keep your branch up to date with changes being made to `main`. To do this you can...
     1. Checkout main --> `git checkout main`
     2. Pull remote changes --> `git pull`
     3. Checkout the feature branch --> `git checkout my_feat_branch`
     4. Merge updated main into feature branch --> `git merge main`
   - Merging may result in merge conflicts. Your IDE will likely guide you through this.
7. Create an upstream/remote copy of the branch and push all commits to it --> `git push --set-upstream origin my_feat_branch`. Going forward, you'll only need to use `git push` on this branch to push any additional commits from the local to the remote.
8. Open PR on GitHub from `my_feat_branch` to `main`.
9. Once approved, click the big green merge button! There are a couple of options for this, but I would recommend `Squash and Merge`
10. Pull most recent changes made to `main` (which now include the changes from the feature branch).
    1. `git checkout main`
    2. `git pull`
11. Cleanup the old feature branch --> `git branch -D my_feat_branch`.
12. Goto step 0.

### Review Workflow

This is the workflow I use when reviewing other people's branches.

0. Currently on `main` branch. Colleague has just opened a PR for their `brand_new_feature` branch.
1. Check for new remote branches --> `git fetch`
2. Checkout their branch --> `git checkout brand_new_feature` (without the -b). Even though no local branch called `brand_new_feature` exists yet, this will create a local branch called `brand_new_feature` that is automatically setup to track `origin/brand_new_feature`.
3. Sanity check: `git pull` to make sure the branch has the latest changes.
4. Run the code and test the branch!
5. Realize that there's a mistake in the code. Make a change request on GitHub.
6. _Colleague fixes the feature branch and pushes there changes_
7. Pull the latest changes `git pull` and test again.
8. Once the PR looks good to go, approve it! It is customary for the opener of the PR to close it.

### Multiple Devs on One Branch

If multiple devs are working on the same feature branch together.

0. Devs John and Jane are working on the `shared_feat_branch` branch.
1. Both John and Jane make changes on their local copies of the branch.
2. Jane finishes her work first and pushes her changes to the remote. John is now blocked from pushing because his branch is behind the remote.
3. John uses `git pull --rebase` to get the changes. This will reorder the commit history of his local branch, so that the changes that Jane made come before his.
   - This may result in rebase conflicts. Your IDE will likely guide you through this.
4. John now pushes his changes to the remote.

### Some Other Useful Stuff

- `origin` = the remote repository that the local project was originally cloned from. It's possible to fork a repo, which is why this could change, but I've never done this personally. Forking is usually only used on public repos, since the owners don't want random people to be able to create random branches. If you are just working off of `origin`, you don't need to explicitly add `origin` to any commands, since it is the default.
- `git fetch` = fetches the current **commit history** of all `origin` branches (including new branches your machine wasn't aware of before). This does NOT pull any changes from branches. You will still need to `git pull` to update your branch. - Fetch is mainly good for checking if there have been changes pushed to a remote branch or discovering new remote branches. - You don't need to run fetch before running pull.
- `git status` = In addition to showing staged and unstaged changes to the working directory, this command will let you know how many commits ahead or behind your local branch is compared to its remote (ex. "Your branch is behind 'origin/main' by 2 commits").
- `git log` = Displays the list of all commits on the current branch. Press `q` to exit.
- `git merge --abort` = if after merging main into your feature branch there are conflicts and for whatever reason you want to cancel the merge instead of resolving the conflicts, this command will return your branch to the state before `git merge branch_name`.
- `git rebase --abort` = same as `git merge --abort` except for when conflicts arise due to a `git pull --rebase`.
