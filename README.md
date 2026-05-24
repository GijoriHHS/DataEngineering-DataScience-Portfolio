# DEDS Portfolio

A coursework portfolio showcasing data science, data engineering, and AI experiments built with Python. The repository includes a Pygame-based Q-learning maze demo, a crewAI lesson-planning agent notebook, and several notebooks for Kaggle, neural networks, and data warehouse exercises.

## 📌 Short Project Summary

This repository is a student portfolio of practical machine learning, reinforcement learning, and data engineering exercises. It is organized as a collection of standalone demos and notebooks rather than a single packaged application.

## 🖼️ Screenshots

### Q-learning Maze — Start State
![Q-learning start](ScreenShots/Start%20Q%20Learning.jpg)
*Initial view of the maze demo in manual or training mode before the agent has completed learning.*

### Q-learning Maze — Learned Policy
![Q-learning final](ScreenShots/Final%20Q%20Learning.jpg)
*Final Q-learning state showing learned values and the agent navigating toward the goal.*

### Data Analysis Workflow — Decision Tree
![Decision Tree](ScreenShots/Decision%20Tree.jpg)
*Example result from the analysis notebooks demonstrating supervised learning model output.*

### Data Analysis Workflow — Confusion Matrix
![Confusion Matrix](ScreenShots/Confusion%20Matrix.jpg)
*Classification report visualization from the repository’s notebook experiments.*

## 🎯 Project Goals

- Demonstrate reinforcement learning with a Pygame maze environment.
- Showcase an AI agent workflow using crewAI for lesson planning.
- Present structured coursework in machine learning, data warehousing, and Python programming.
- Provide a visual portfolio of models, datasets, and engineering exercises.

## ✨ Features

- Interactive Q-learning maze with both manual control and autonomous training.
- Visual display of learned Q-values alongside the maze grid.
- crewAI agent workflow for generating lesson plans and content using an LLM-backed multi-agent approach.
- Notebook-based data science exercises for Kaggle data, neural networks, and data warehouse concepts.

## 🧩 Technologies Used / Tech Stack

- Python 3.10+ (project configured for `>=3.10,<3.13`)
- `numpy`
- `pandas`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyodbc`
- `pygame`
- `dotenv`
- `crewai` (used in the AI agent demo notebook)

> Note: `crewai` and `pygame` are referenced by the repository code but are not currently declared in `pyproject.toml`.

## 🔧 How the Application Works

### Q-learning Maze Demo

The `PyGame_Project/Game.py` and `PyGame_Project/GridGame.py` scripts implement a simple grid-based maze environment:

- Maze tiles represent walls, hazards, and a goal state.
- The player can move manually with arrow keys.
- Press `T` to switch to training mode, where the Q-learning agent learns the best path over episodes.
- The simulation renders both the maze and the agent’s Q-values in real time.
- Controls include training, manual play, reset, and speed adjustment.

### crewAI Lesson Planning Demo

The root-level `Leer Ai Agent.py` script demonstrates a crewAI workflow using:

- `Agent` objects for planner, explainer, and reviewer roles.
- `Task` objects that define lesson planning, content creation, and review steps.
- `Crew` to coordinate sequential execution and generate markdown-style lesson output.

This demo is designed for interactive notebook execution rather than a standalone CLI app.

## 📂 Data Architecture / Data Handling

This repository includes datasets used in notebooks, but the main runnable scripts do not directly ingest these CSV files.

Data files included:

- `Kaggle/vgsales.csv`
- `DWH_Group/inventory_levels_train.csv`
- `DWH_Group/product_forecast_train.csv`
- `SuperVised, Unsupervised L_ng/titanic.csv`

The data exploration and model training details are primarily contained in the notebook files across the repository.

## 📁 Project Structure

```
.
├── pyproject.toml
├── README.md
├── utils.py
├── Leer Ai Agent.py
├── ScreenShots/
│   ├── Start Q Learning.jpg
│   ├── Final Q Learning.jpg
│   ├── Decision Tree.jpg
│   └── Confusion Matrix.jpg
├── PyGame_Project/
│   ├── Game.py
│   ├── GridGame.py
│   └── Sprites/
├── AI_Agent/
│   └── CrewAI.ipynb
├── DWH_Group/
│   ├── DataWareHouse.ipynb
│   ├── inventory_levels_train.csv
│   └── product_forecast_train.csv
├── Kaggle/
│   ├── Kaggle_1.ipynb
│   ├── Kaggle_2.ipynb
│   └── vgsales.csv
├── Neural_Network/
├── Python_Introduction/
├── SuperVised, Unsupervised L_ng/
└── src/deds_portfolio/
    └── __init__.py
```

### Notes on structure

- `PyGame_Project/` contains the runnable reinforcement learning demo.
- `AI_Agent/` contains the crewAI notebook demo.
- `DWH_Group/`, `Kaggle/`, `Neural_Network/`, and similar folders contain coursework notebooks and datasets.
- `src/deds_portfolio/` is a package placeholder with no implementation.

## 🚀 Setup / Installation / Running the Project

1. Install dependencies using PDM or pip.

```bash
pdm install
```

2. Install or verify additional packages that may be required:

```bash
pip install pygame crewai python-dotenv
```

3. Run the Q-learning demo:

```bash
python PyGame_Project/Game.py
```

4. Run the AI agent demo in a notebook environment:

- Open `Leer Ai Agent.py` or `AI_Agent/CrewAI.ipynb` in Jupyter.
- Ensure your `.env` file contains an `OPENAI_API_KEY` if you want to execute the OpenAI-backed agent flow.

## ⚠️ Notes / Limitations

- This repository is a portfolio of coursework, so many folders contain notebooks rather than polished production code.
- `src/deds_portfolio` and `tests/__init__.py` are present as placeholders without active implementation.
- `GridGame.py` largely duplicates `Game.py`; the repository contains two similar maze demo scripts.
- The dataset CSV files are available, but the core Python scripts do not automatically load them.

## 📜 License / Academic Note

- `pyproject.toml` declares the project license as MIT.
- This repository is presented as an academic / portfolio showcase of data science and AI course work.
