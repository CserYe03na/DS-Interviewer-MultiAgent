# DS Interview Preparation Assistant

An intelligent multi-agent planning system that assists candidates in preparing for data science technical interviews. Given the days left, a job description, and a user profile, it can generate a **skill-aware**, **difficulty-controlled**, and **time-constrained** study plan.

---

## 🌟 Features

- **Multi-Agent Planning Architecture**  
A system composed of specialized agents that collaboratively analyze job requirements, retrieve relevant interview problems, and generate structured preparation plans.

- **Skill-Aware Interview Preparation**  
Identifies required skills from job descriptions and user profile, and allocates preparation effort accordingly.

- **Retrieval with Post-Planning Allocation**  
Retrieves a large candidate pool of interview problems using taxonomy skills as queries, then enforces joint skill and difficulty constraints during later planning.

- **Structured Study Plans**  
Generates executable interview preparation plans based on user preference (e.g., 7 / 14 / 30-day schedules).

## 🏗️ Architecture

### Agent 1 — Skill Extraction & Weighting

* `scripts/Agent1/input_analyzer.py`: Extracts data-science technical skill signals from the job description and the user's self-description using an LLM.
* `scripts/Agent1/skill_mapper.py`: Maps extracted skill keywords to the project's predefined taxonomy skills via an LLM-assisted matching step.
* `scripts/Agent1/weight_allocator.py`: Assigns a weight to each taxonomy skill based on frequency and requirement strength (e.g., must-have vs. preferred).
* `scripts/Agent1/skill_analyzer_agent.py`: Orchestrates the Agent 1 pipeline and outputs the final skill–weight profile.

### Scope Planner — Global Plan Constraints

* `scripts/scope_planner_agent.py`: Determines the overall preparation scope, i.e. total questions, difficulty distribution, and per-skill quotas, given skill weights, time budget, and user constraints.

### Agent 2 — Retrieval

* `scripts/Agent2/retrieval.py`: Retrieves relevant interview questions using embedding-based semantic search.

### Agent 3 — Planning & Scheduling

* `scripts/Agent3/Planning_Agent.py`: Generates a day-by-day study plan by scheduling retrieved tasks under joint constraints (difficulty curve, skill balance) with LLM-based refinement and summaries.

## 🔧 Reproducible workflow

### **1. Data Pipeline**

#### **a. SQL Leetcode Database**

* **Data Source**
  * The raw SQL LeetCode dataset is exported from the public repository: `https://github.com/mrinal1704/SQL-Leetcode-Challenge/blob/master/` 
  * The exported raw files are stored under: `data/sql_raw`

* **Schema Standardization**
  * `data_prep/extract_sql_raw.py` converts each problem into the project standardized JSON schema and writes the result to `data/sql_raw_extracted.json`
  * Each record follows the unified schema:
    
    ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "SQL",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":"<github link>"
        }
    }
    ```

* **Taxonomy Annotation**
  * `data_prep/annotate.py` uses an LLM to annotate/tag each SQL problem with the pre-defined taxonomy skills
  * The taxonomy definition is stored in: `data/taxonomy_skills.json`
  * The annotated SQL dataset is written to: `data/sql.json`


#### **b. Algorithms/Pandas Leetcode Database**
* **Data Source**
  * The raw Algorithms and Pandas LeetCode dataset is exported from Kaggle: `https://www.kaggle.com/datasets/alishohadaee/leetcode-problems-dataset`
  * The exported raw file is stored under: `data/leetcode_problems_raw.json`, and `data/algorithms_raw.json` is extracted from this file
* **Schema Standardization**
  * `data_prep/algo_preprocessing.py` and `data_prep/pandas_preprocessing.py` converts each problem into the project standardized JSON schema and annotate the taxonomy skills, finally writes the results to `data/algorithms.json` and `data/pandas.json` respectively
  * Each record follows the unified schema based on category:
      ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "Algorithms",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":""
        }
    }
    ```
    ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "Pandas",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":""
        }
    }
    ```
    

#### **c. Leetcode Dataset Merge**

  * `data_prep/merge_leetcode.py` merges the SQL dataset with the Python/algorithm LeetCode dataset, aligning both to a unified schema. The merged output is written to: `data/leetcode.json`

#### **d. Theory Database**

* **Data Source**

  * Theory Q&A content is stored as markdown files under: `data/TheoryQA/` (all `*.md` files).

* **Extraction & Cleaning**

  * `data_prep/theory_extraction_and_cleaning.py` parses markdown sections and Q&A blocks, performs text cleaning/normalization, and repairs taxonomy/subdomains when needed.
  * Extraction rules:

    * Section header (`## ...`) is treated as a **subdomain** candidate.
    * Q&A pairs are extracted from `**Question**`-style headings and subsequent lines until the next question/section.

* **Output Format**

  * The processed dataset is written as JSONL to: `data/Theory.jsonl`.
  * Each record follows the unified schema:

    ```json
    {
      "id": "",
      "vector_content": "<cleaned answer text>",
      "metadata": {
        "title": "<normalized question>",
        "domain": "theory",
        "subdomain": "<normalized/routed subdomain>",
        "taxonomy_skill": ["<subdomain>"],
        "raw_topics": [],
        "source": "<markdown filename>"
      }
    }
    ```

* **Taxonomy/Subdomain Repair**

  * If the extracted subdomain is missing/too generic, the script assigns a more specific subdomain using keyword-based routing (e.g., `sql`, `databases`, `clustering`, `fairness_and_imbalance`).


### **2. Local Installation & Launch**

- Clone and enter the project
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_PROJECT_FOLDER>
```

- To configure environment variables, create a local `.env` from the template:
```bash
cp .env.example .env
```

- Open `.env` and fill in your key:
```bash
OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
```

- Create the conda environment for package installation (first-time setup)
```bash
conda env create -f environment.yml
conda activate ds-interviewer
```

- Run Streamlit by `python -m` to ensure Streamlit runs under the correct conda environment:
```bash
python -m streamlit run demo.py
```
## 💡 Usage Example
demo url: <<FILL_IN>>
