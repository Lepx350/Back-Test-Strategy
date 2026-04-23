# Deploy to Railway ☁️

**Goal:** Get your backtest dashboard running in the cloud, accessible from any phone browser anywhere.

**Time:** 20 minutes
**Cost:** $5/month (Railway Starter) after $5 free trial

---

## Prerequisites

- [ ] GitHub account ([github.com](https://github.com) — free)
- [ ] Railway account ([railway.app](https://railway.app) — sign in with GitHub)
- [ ] Git installed on PC ([git-scm.com](https://git-scm.com))

---

## Step 1: Create GitHub Repo (5 min)

1. Go to github.com → **New repository**
2. Name it: `volman-cloud` (or whatever you want)
3. Set to **Private** (keeps your data safe)
4. Don't initialize with README
5. Click Create

---

## Step 2: Push Your Code (5 min)

Open PowerShell in `D:\volman_complete\python_backtest\` (or wherever the cloud folder ended up):

```powershell
# Initialize git
git init

# Add all files (except what's in .gitignore)
git add .

# Commit
git commit -m "Initial cloud deploy"

# Link to your GitHub repo (replace YOUR-USERNAME and REPO-NAME)
git remote add origin https://github.com/YOUR-USERNAME/volman-cloud.git
git branch -M main
git push -u origin main
```

First push will ask you to log into GitHub. Use a **Personal Access Token** (not your password):
1. GitHub → Settings → Developer settings → Personal access tokens → Generate new
2. Give it `repo` scope
3. Use that token as your password when pushing

---

## Step 3: Deploy to Railway (5 min)

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo**
3. Select your `volman-cloud` repo
4. Railway detects Python, starts building

**Wait ~3 minutes for first deploy.**

---

## Step 4: Configure Railway (2 min)

Once deployed, click your service → **Settings**:

1. **Generate Domain** — Railway gives you a URL like `yourapp.up.railway.app`
2. Copy that URL — this is your dashboard!

Optional but recommended:
- **Environment Variables** tab → Add `DATABENTO_API_KEY = db-xxxx` (if you want to fetch data from the cloud app)

---

## Step 5: Upload Your Data (3 min)

Your ES data needs to reach the cloud. Two options:

### Option A: Commit data to GitHub (easiest if < 100MB)

```powershell
# Back in your local project folder
# Remove data/ from .gitignore first (edit .gitignore, delete the "data/*.parquet" line)

git add data/
git commit -m "Add ES historical data"
git push
```

Railway auto-redeploys. Data is now available.

⚠️ GitHub has a **100MB file size limit.** If your parquet is bigger, use Option B.

### Option B: Use Git LFS (for big files)

```powershell
# Install git-lfs from git-lfs.com, then:
git lfs install
git lfs track "*.parquet"
git add .gitattributes
git add data/
git commit -m "Add data via LFS"
git push
```

### Option C: Upload manually (Railway CLI)

```powershell
npm install -g @railway/cli
railway login
railway link  # select your project
railway run --service backend bash
# Then upload files via SFTP or re-fetch from Databento
```

---

## Step 6: Open on Your Phone 📱

1. On phone, open browser
2. Go to your Railway URL (`yourapp.up.railway.app`)
3. **Bookmark it** or add to home screen
4. Tap, run backtests, see results

On iPhone: Safari → Share → Add to Home Screen → acts like an app

On Android: Chrome → menu → Add to Home Screen

---

## Updating Your App

Any time you want to change code:

```powershell
# Make changes locally
git add .
git commit -m "What I changed"
git push

# Railway auto-deploys in ~2 minutes
```

---

## Troubleshooting

**"Application failed to respond"**
- Check Railway logs (Dashboard → your service → Logs)
- Usually a missing package — check `requirements.txt`

**"No data found"**
- Data folder didn't upload — check Step 5

**App is slow**
- Starter plan has 512MB RAM. Upgrade to Hobby ($10/mo) if needed.
- Or: use smaller data slices (1 year instead of 6)

**Can't log in to GitHub from terminal**
- Use Personal Access Token instead of password
- Or use GitHub Desktop ([desktop.github.com](https://desktop.github.com)) for GUI

---

## Cost Breakdown

| Plan | Cost | What You Get |
|---|---|---|
| **Trial** | $5 free | First 500 hours |
| **Hobby** | $5/mo | 8 GB RAM, unlimited hours, perfect for this |
| **Pro** | $20/mo | More resources (overkill for now) |

**For your use:** Starter $5/mo is perfect. The app runs 24/7, sleeps when unused, wakes in seconds.

---

## What You've Built 🎉

One URL. Any device. Tap to run backtests.

- Your PC: optional now
- Phone: full capability
- Pipeline: 100% cloud-hosted
- Your edge: accessible anywhere

**Welcome to cloud quant research.** 🚀
