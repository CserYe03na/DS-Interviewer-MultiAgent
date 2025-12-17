# DS Interview Preparation Assistant

An intelligent multi-agent planning system that assists candidates in preparing for data science technical interviews by generating **skill-aware** and **difficulty-controlled** interview preparation plans.

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

```
retrieval.py            # Retrieves relevant interview questions using embedding-based semantic search
Planning_Agent.py       # Generates a study plan by allocating retrieved tasks across days using difficulty curves, skill balance, and LLM-based refinement and summaries.
```

## Reproducible workflow

- **Data Pipeline**

- SQL Leetcode Database

1. Data Source
- The raw SQL LeetCode dataset is exported from the public repository: `https://github.com/mrinal1704/SQL-Leetcode-Challenge/blob/master/` 

- The exported raw files are stored under: `data/sql_raw`

2. Schema Standardization
- `data_prep/extract_sql_raw.py` converts each problem into the project standardized JSON schema and writes the result to `data/sql_raw_extracted.json`

Target JSON schema
{
    "vector_content": "",
    "metadata": {
    “id”:
    "title": "",
    "category": "",
    "taxonomy_skill": [ ],
    "solution_summary": "",
    "url": ""，
    “backup_url”:””
    }
}

3. Taxonomy Annotation
- `data_prep/annotate.py` uses an LLM to annotate/tag each SQL problem with the pre-defined taxonomy skills

- The taxonomy definition is stored in: `data/taxonomy_skills.json`

- The annotated SQL dataset is written to: `data/sql.json`


- Python/Pandas Leetcode Database

- Leetcode Dataset Merge
`data_prep/merge_leetcode.py` merges the SQL dataset with the Python/algorithm LeetCode dataset, aligning both to a unified schema. The merged output is written to: `data/leetcode.json`

- Theory Database

- **Multi-Agent Local Run**

0. Clone and enter the project
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_PROJECT_FOLDER>
```

1. Configure environment variables
- Create a local `.env` from the template:
```bash
cp .env.example .env
```

- Open `.env` and fill in your key:
```bash
OPENAI_API_KEY="TYPE_YOUR_OPENAI_KEY_HERE"
```

2. Create the conda environment for package installation (first-time setup)
```bash
conda env create -f environment.yml
conda activate ds-interviewer
```

3. Run Streamlit
Use `python -m` to ensure Streamlit runs under the correct conda environment:
```bash
python -m streamlit run streamlit.py
```