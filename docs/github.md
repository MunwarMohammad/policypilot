# Push this project to your GitHub

The repository files contain configuration examples, not credentials. `.env`, local data,
reference materials, build outputs, caches and virtual environments are excluded.

Create an empty repository on GitHub, then run from this project directory:

```bash
# Run init only if this folder is not already a repository.
git init -b main
git add .
git diff --cached --stat
git status
# Review the staged files before committing.
git commit -m "Build PolicyPilot agentic RAG application"
git remote add origin https://github.com/YOUR_USERNAME/policypilot.git
git push -u origin main
```

Replace YOUR_USERNAME with your actual account. If `origin` already exists, inspect it
with `git remote -v` rather than adding another. GitHub authentication uses your normal
Git credential manager or `gh auth login`; never put tokens in a remote URL.

CI runs offline tests, linting and package build. Cloud deployment is manual and needs
separate setup. Add a repository description such as:

> SEC filing research application with LangGraph, citation checks, semantic retrieval,
> and Azure Databricks deployment adapters.

Do not describe it as deployed, evaluated across 20 companies, or enterprise production
ready until those activities have actually been completed. Choose a software license
before inviting external reuse; no license has been selected on your behalf.
