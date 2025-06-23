## 💻 Github Commands

### Initial Setup
```bash
cd /Users/romainjouen/Documents/WORK/DEV/portfolio-ai

git init
git add .
git commit -m "Initial commit"
#Create a new repo on GitHub called "portfolio-ai"
git remote add origin https://github.com/romainjouen/portfolio-ai.git 
git branch -M main
git push -u origin main
```

### Updates
```bash
git status #Check what's changed
git add . #Add all changes
# OR add specific files
git add README.md .gitignore
git commit -m "Update README and add .gitignore file"
git push
#If you're on a branch other than main
git push origin <your-branch-name> 
```

### File Management
```bash
# Remove files from Git but keep locally
git rm --cached <file_name>
git commit -m "Remove <file_name> from the repository"
git push
```

### Clone Project
```bash
# Retrieve the project from Github
git clone https://github.com/romainjouen/local-swarm-sql-agents.git
cd local-swarm-sql-agents

# Set up your virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```