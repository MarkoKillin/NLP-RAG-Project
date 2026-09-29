# Git Basics

## What is Git?

Git is a distributed version control system that tracks changes to files over
time. It was created by Linus Torvalds in 2005 to manage development of the
Linux kernel. Unlike centralized systems, every developer holds a full copy of
the project history on their own machine, so most operations are local and fast.
Git records the state of a project as a series of snapshots and lets you move
between them, compare them, and combine work from many people.

## Core concepts

- Repository. A repository (repo) is the project and its complete history,
  stored in a hidden `.git` directory. It holds every commit, branch, and tag.
- Commit. A commit is a snapshot of the tracked files at a point in time, along
  with a message, author, and a pointer to its parent commit. Each commit has a
  unique SHA-1 hash that identifies it.
- Branch. A branch is a movable pointer to a commit. It lets you develop a
  feature or fix in isolation without affecting other lines of work. The default
  branch is commonly named `main` or `master`.
- Merge. Merging combines the changes from one branch into another, producing a
  new commit that has two parents when the histories have diverged.
- Remote. A remote is a version of the repository hosted elsewhere, such as on
  GitHub or a company server. `origin` is the conventional name for the default
  remote. You sync with it using push and pull.
- Staging area. Also called the index, the staging area is where you assemble
  the exact set of changes that will go into the next commit. Files move from
  the working directory to the staging area, then into a commit.
- HEAD. HEAD is a pointer to the current commit you are working on, usually the
  tip of the checked-out branch. It determines what your working directory
  reflects and what the next commit will build on.

## Common commands

- `git init`. Create a new empty repository in the current directory.
  Usage: `git init`
- `git clone`. Copy an existing repository, including its history, from a
  remote. Usage: `git clone <url>`
- `git add`. Stage changes for the next commit.
  Usage: `git add <file>` or `git add .`
- `git commit`. Record the staged changes as a new commit.
  Usage: `git commit -m "message"`
- `git status`. Show which files are staged, modified, or untracked.
  Usage: `git status`
- `git log`. Display the commit history.
  Usage: `git log --oneline`
- `git branch`. List, create, or delete branches.
  Usage: `git branch <name>` to create, `git branch` to list
- `git checkout` / `git switch`. Move to another branch or restore files.
  `switch` is the newer, branch-focused command.
  Usage: `git switch <branch>` or `git checkout -b <new-branch>`
- `git merge`. Combine another branch into the current one.
  Usage: `git merge <branch>`
- `git rebase`. Reapply commits from the current branch on top of another base.
  Usage: `git rebase <branch>`
- `git pull`. Fetch changes from a remote and merge them into the current
  branch. Usage: `git pull origin main`
- `git push`. Send local commits to a remote branch.
  Usage: `git push origin <branch>`
- `git stash`. Temporarily shelve uncommitted changes and revert to a clean
  working directory. Usage: `git stash`, then `git stash pop` to restore
- `git reset`. Move the current branch to a different commit and optionally
  change the staging area and working tree.
  Usage: `git reset --hard <commit>` or `git reset <file>` to unstage
- `git revert`. Create a new commit that undoes the changes of a previous
  commit. Usage: `git revert <commit>`

## Merge vs. rebase

Both merge and rebase integrate work from one branch into another, but they do
it differently. Merge creates a new merge commit that joins the two histories
and keeps their original order intact, so the record shows exactly when and how
branches came together. Rebase instead moves your commits so they replay on top
of the target branch, producing a linear history with no merge commit. Merge
preserves history as it happened; rebase rewrites it to look cleaner. Because
rebase changes commit hashes, avoid rebasing commits that others have already
pulled.

## Reset vs. revert

Reset and revert both undo changes but affect history differently. Reset moves
the branch pointer backward and can discard commits, which rewrites history and
is best used on local, unpushed work. Revert leaves the existing history in
place and adds a new commit that reverses the effect of an earlier commit, which
makes it safe for changes already shared with others.