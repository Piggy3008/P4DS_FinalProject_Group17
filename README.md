# Group 17 — Programming for Data Science Final Project

WA-EV-Insights explores the Washington State Electric Vehicle (EV) Population dataset for **CSC17104 – Programming for Data Science**. Notebooks 01 and 02 cover source and data-quality review, exploratory analysis, and research questions Q1–Q5. Notebook 03 contains the separate Q6 machine-learning work.

## Repository structure

```text
README.md
environment.yml
LLM_USAGE.md
slides.pdf
data/
  raw/                  
  processed/            
notebooks/
  01_collection_exploration.ipynb
  02_analysis_q1_q2.ipynb
  02_analysis_q3_q4.ipynb
  02_analysis_q5.ipynb
  03_ml_question.ipynb  # Q6
src/
  cleaning.py
  features.py
  stats_utils.py
  viz.py
slides/
  Group17_FinalProject.ppt
```

## Members and ownership

| Member | Student ID | Primary ownership |
|---|---:|---|
| Nguyễn Trực Phúc | 24127109 | Data source/Data Card, `src/cleaning.py`; Q1–Q2 |
| Phạm Tấn Nhật Thịnh | 24127244 | Data quality/EDA numericals, `src/viz.py`; Q3–Q4 |
| Nguyễn Thanh Toàn | 24127559 | Policy/ML readiness review; Q5 and Q6 (ML) |

## Datasets

The primary dataset is downloaded from [Electric Vehicle Population Data 2024 on Kaggle](https://www.kaggle.com/datasets/utkarshx27/electric-vehicle-population-data/data), uploaded by Utkarsh Singh (2024 snapshot). The upstream administrative source is the **Washington State Department of Licensing (DOL)** via [Data.WA.gov (`f6w7-q2d2`)](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2) and [Data.gov](https://catalog.data.gov/dataset/electric-vehicle-population-data), released under the **Open Data Commons Open Database License (ODbL) v1.0** for public and educational use.

| Role | Local file | Observation grain |
|---|---|---|
| Main | `data/raw/Electric_Vehicle_Population_Data.csv` | One registered BEV or PHEV per row (177,866 rows, 17 columns) |
| Auxiliary (for Q1 / D1) | `data/raw/wa_county_demographics.csv` | One Washington county per row (population and median household income from U.S. Census / WA OFM) |

Download `Electric_Vehicle_Population_Data.csv` from the Kaggle link above and place it in `data/raw/`. The notebooks read raw files without modifying them. Keep raw CSVs and downloaded archives local; `data/raw/` and `data/processed/` are ignored by Git.

Key data-quality notes documented in Notebook 01:
- `Electric Range == 0` appears in 91,950 rows (51.7%), primarily reflecting newer BEVs whose battery range has not yet been researched by DOL rather than a true 0-mile range.
- `Base MSRP == 0` appears in 174,522 rows (98.1%) and is treated as unrecorded/missing rather than a $0 price.
- `Electric Utility` contains pipe-separated (`|`) multi-utility strings for some registrations; Q2 explicitly tests sensitivity to how these multi-utility records are assigned.

### Existing public work

Because this is a widely used public dataset, several public notebooks exist on the [Kaggle Code tab](https://www.kaggle.com/datasets/utkarshx27/electric-vehicle-population-data/code) (16 public notebooks on this mirror) as well as external analyses such as [*The Status of Electric Vehicles in Washington*](https://merjemmm.github.io/WA-EV-Analysis/). Most public notebooks focus on descriptive top-10 bar charts (top makes, top models, raw county counts) or basic range histograms. In contrast, Q1–Q5 in this project normalize adoption by county population/income (D1/D3), control for confounders such as urban/rural status and BEV/PHEV type (D4), quantify uncertainty with bootstrap confidence intervals and effect sizes (D5), and run explicit robustness checks on `Electric Range == 0` and multi-utility strings (D6). Q6 frames a leakage-free, time-split classification task for infrastructure planning rather than a random train/test split. Before submission, the group will verify and list any additional Kaggle notebooks reviewed.

## Environment and run order

From the repository root, create and activate the declared Conda environment:

```bash
conda env create -f environment.yml
conda activate wa-ev-insights
python -m jupyter lab
```

Run notebooks in this order so that source checks and preprocessing assumptions are reviewed before analysis:

1. `notebooks/01_collection_exploration.ipynb`
2. `notebooks/02_analysis_q1_q2.ipynb`
3. `notebooks/02_analysis_q3_q4.ipynb`
4. `notebooks/02_analysis_q5.ipynb`
5. `notebooks/03_ml_question.ipynb` for Q6, after the group reviews its ML design.

The notebooks load CSVs from `data/raw/` and import shared helper functions from `src/` (`cleaning.py`, `features.py`, `stats_utils.py`, `viz.py`). All analysis choices, row-count checks, tables, plots, and interpretations remain visible in the notebooks.

## EDA snapshot and remaining group work

The main file contains **177,866** registered EV rows spanning Model Years **1997–2024**: **139,210** Battery Electric Vehicles (BEVs, 78.3%) and **38,656** Plug-in Hybrid Electric Vehicles (PHEVs, 21.7%). Missingness is low in standard null counts (`Legislative District`: 389 nulls; `Vehicle Location`: 9 nulls; `County`/`City`/`Postal Code`/`Electric Utility`/`2020 Census Tract`: 5 nulls each), but placeholder zeros in `Electric Range` (91,950) and `Base MSRP` (174,522) require explicit handling. The EDA and analysis notebooks report row counts before and after every filter or merge and include uncertainty/robustness checks where required.

## Research questions

| ID | Question | Owner |
|---|---|---|
| Q1 | How have the technological composition (BEV vs. PHEV) and Electric Range trends evolved across Model Year generations? | Nguyễn Trực Phúc |
| Q2 | How do household income and public charging station density affect electric vehicle adoption rates across regions (County / 2020 Census Tract) when controlling for urban-rural classification? | Nguyễn Trực Phúc |
| Q3 | Does geographic distance to the economic hub of Seattle influence consumer preference between BEVs and PHEVs? | Nguyễn Thanh Toàn |
| Q4 | How has market share concentration (oligopoly) between Tesla and legacy automakers shifted across model year cycles within Legislative Districts? | Nguyễn Thanh Toàn |
| Q5 | Do disparities in electric vehicle adoption across Electric Utilities reflect service-area demographics or structural differences between Public Utility Districts (PUDs) and Investor-Owned Utilities (IOUs)? | Phạm Tấn Nhật Thịnh |
| Q6 | How accurately can geographic location, power utility infrastructure, and regional socioeconomic context predict whether newly registered owners choose a battery electric vehicle (BEV) over a plug-in hybrid electric vehicle (PHEV)? | Phạm Tấn Nhật Thịnh |
