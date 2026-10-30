# Beginner upload instructions

This folder has already been prepared as a sanitized public repository.

## 1. Review the files

Open `README.md` and confirm the description is accurate. Review `LICENSE-CHOICES.md` and decide whether you want others only to view the project or also to reuse it.

## 2. Sign in to GitHub from the terminal

Run:

```bash
gh auth login
```

Choose:

1. `GitHub.com`
2. `HTTPS`
3. `Login with a web browser`

The terminal will display a one-time code. Copy it, press Enter, and approve access in the browser. Do not send that code or an access token to anyone.

## 3. Set your Git author identity

Replace the example values with your real name and the email address connected to GitHub:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
git commit -m "Initial public research release"
```

The files have already been staged, so you do not need to run `git add .` for the first commit.

## 4. Create and upload the repository

From this folder, run:

```bash
gh repo create mental-health-safety-policy --public --source=. --remote=origin --push
```

If the repository name is already taken, choose another name, such as `mental-health-chatbot-safety-research`.

## 5. Open the repository

Run:

```bash
gh repo view --web
```

## 6. Future updates

After editing files:

```bash
git status
git add PATH_TO_THE_FILE_YOU_CHANGED
git commit -m "Describe the update"
git push
```

Avoid `git add .` until you have reviewed `git status`, especially if you add local data files.

