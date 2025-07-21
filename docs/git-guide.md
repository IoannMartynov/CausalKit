# How to Commit and Push a Branch in Git

## What is a Git Branch?
A branch in Git is a lightweight movable pointer to a commit. It represents an independent line of development, allowing you to work on features or fixes without affecting the main codebase until you're ready to merge your changes.

## Check your current branch and status
```bash
git branch        # Shows all local branches with current branch marked with *
git status        # Shows modified files and staging status
```

## Stage all changes
```bash
git add .         # Stages all changes in the current directory and subdirectories
git add -A        # Stages all changes (including deletions) in the entire repository
```

## Commit your changes
```bash
git commit -m "Your descriptive commit message here"
```

You can also stage and commit in one step (only for tracked files):
```bash
git commit -am "Your descriptive commit message here"
```

## Push your branch to the remote repository
If the branch already exists on the remote:
```bash
git push         # Pushes current branch to its upstream remote
```

If this is a new branch that doesn't exist on the remote yet:
```bash
git push -u origin your-branch-name   # Creates remote branch and sets upstream
```

Replace `your-branch-name` with the name of your current branch.

## Additional Useful Git Commands

### Branch Management
```bash
git checkout -b new-branch-name       # Create and switch to a new branch
git checkout existing-branch          # Switch to an existing branch
git branch -d branch-name             # Delete a branch locally
git push origin --delete branch-name  # Delete a branch on the remote
```

### Syncing with Remote
```bash
git fetch                             # Download objects and refs from remote
git pull                              # Fetch and integrate changes from remote
git merge origin/branch-name          # Merge a remote branch into your current branch
```

### Viewing History
```bash
git log                               # View commit history
git log --oneline                     # View simplified commit history
git diff                              # View unstaged changes
git diff --staged                     # View staged changes
```

## Best Practices

1. **Pull before pushing**: Always `git pull` before pushing to avoid conflicts.
2. **Commit frequently**: Make small, focused commits with clear messages.
3. **Use descriptive branch names**: Name branches according to what they contain (e.g., `feature/user-authentication`, `bugfix/login-error`).
4. **Keep branches up to date**: Regularly merge or rebase with the main branch.
5. **Delete merged branches**: Clean up branches after they've been merged.
6. **Review changes before committing**: Use `git status` and `git diff` to review changes.