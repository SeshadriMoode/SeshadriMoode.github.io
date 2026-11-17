@echo off
REM 1) Create empty repo at https://github.com/new  (name: pacc-platoon-visualizer, public)
REM 2) Replace YOUR_GITHUB_USERNAME below, then run this script once.

set GH_USER=YOUR_GITHUB_USERNAME
set REPO=pacc-platoon-visualizer

git remote remove origin 2>nul
git remote add origin https://github.com/%GH_USER%/%REPO%.git
git push -u origin main

echo.
echo After push: GitHub repo -^> Settings -^> Pages -^> Source: Deploy from branch main, folder / (root)
echo Live URL: https://%GH_USER%.github.io/%REPO%/
