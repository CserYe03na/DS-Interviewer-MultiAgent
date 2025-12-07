import json
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

IN_PATH = Path("data/sql_raw_extracted.json")
OUT_PATH = Path("data/sql.json")

client = OpenAI()

MODEL = "gpt-4o-mini"

# sql taxonomy
SQL_SKILL_TAXONOMY: List[str] = [
    "sql_basic",    
    "sql_filter",    
    "sql_aggregation", 
    "sql_group_by",
    "sql_sorting",
    "sql_distinct",
    "sql_join",
    "sql_subquery",
    "sql_window_function",
    "sql_case_when",
    "sql_date_processing",
    "sql_string_processing",
    "sql_set_operations",
    "sql_cte",
    "sql_null_handling",
    "sql_regexp",
    "sql_pivot"
]


def build_prompt(title: str, vector_content: str, solution_raw: str) -> str:
    skills_str = ", ".join(SQL_SKILL_TAXONOMY)
    template = """
        You are a SQL interview coach assistant.

        You are given a SQL interview coding problem:

        Title: {title}

        Description:
        \"\"\"
        {vector_content}
        \"\"\"

        Here is the **actual SQL solution** to this problem:
        \"\"\"
        {solution_raw}
        \"\"\"

        Your tasks:

        1. From the SQL skill candidate list below, select **1–4 SQL skill tags** that best describe the skills required to solve this problem:
        [{skills_str}]

        ==================== RULES FOR TAG SELECTION ====================

        (1) **STRICT RULE FOR sql_basic**
        Choose **sql_basic ONLY IF** the solution uses mainly:
        - SELECT
        - FROM
        - simple WHERE filtering  
        and **does NOT include ANY** of the following:   
        - subqueries (IN / EXISTS / ANY / ALL)  
        - window functions  
        - CTEs (WITH …)  
        - CASE WHEN  
        - any nontrivial string/date processing  
        - set operations (UNION / INTERSECT / EXCEPT)

        If **any** of the above appear in the SQL solution,  
        → **DO NOT select sql_basic**.

        This rule ensures sql_basic is used *only* for extremely simple queries.

        ---------------------------------------------------------------

        (2) **Rules for string processing (sql_string_processing)**  
        Select **sql_string_processing** if the solution uses ANY of the following:
        - LIKE, NOT LIKE  
        - UPPER(), LOWER(), TRIM(), LENGTH()  
        - SUBSTRING(), LEFT(), RIGHT()  
        - REPLACE(), REGEXP  
        - CONCAT, string concatenation  
        - other explicit string manipulation

        ---------------------------------------------------------------

        (3) **General tag selection rules**
        - Select tags ONLY from the allowed skill list.
        - Prefer more specific skills when applicable.
        - A tag should only be selected if it plays a meaningful or major role in solving the problem.
        - Do NOT overpredict: irrelevant tags must NOT be included.

        ===============================================================

        2. Provide a concise **2–6 sentence English summary** of the solution approach:
        - Summarize the main reasoning **directly from the actual SQL solution shown above**.
        - 1-3 sentences describe how the query works.
        - Optional: 1-3 sentences explain why the approach is correct or efficient.

        ===============================================================

        Your output MUST be valid JSON with exactly two keys:
        - "extra_taxonomy_skills": an array of strings
        - "solution_summary": a string

        Example output:
        {{
        "extra_taxonomy_skills": ["sql_join", "sql_date_processing"],
        "solution_summary": "Join the tables on user_id and apply date filtering to isolate the matching events."
        }}

        Respond ONLY with a valid JSON object.  
        Do NOT include explanations, commentary, or formatting beyond the JSON itself.
        """

    return template.format(
            title=title,
            vector_content=vector_content,
            solution_raw=solution_raw,
            skills_str=skills_str
        )


def call_llm_for_record(title: str, vector_content: str, solution_raw: str) -> Dict:
    prompt = build_prompt(title, vector_content, solution_raw)

    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0.1,
    )

    text = resp.output[0].content[0].text
    try:
        data = json.loads(text)
    except Exception:
        print("[WARN] LLM output not valid JSON, fallback empty; raw:", text[:200])
        data = {
            "extra_taxonomy_skills": [],
            "solution_summary": "",
        }
    data.setdefault("extra_taxonomy_skills", [])
    data.setdefault("solution_summary", "")
    return data


def merge_taxonomy_skills(existing: List[str], extra: List[str]) -> List[str]:
    merged = existing[:] if existing else []
    for tag in extra:
        if tag not in merged:
            merged.append(tag)
    return merged

def annotate_one_problem(record: Dict) -> Dict:
    vector_content = record.get("vector_content", "")
    solution_raw = record.get("solution_raw", "")
    metadata = record.get("metadata", {})

    title = metadata.get("title", "")
    existing_skills = metadata.get("taxonomy_skills", [])

    # LLM call
    llm_result = call_llm_for_record(title, vector_content, solution_raw)
    extra_skills = llm_result.get("extra_taxonomy_skills", [])
    solution_summary = llm_result.get("solution_summary", "").strip()

    # merge taxonomy
    metadata["taxonomy_skills"] = merge_taxonomy_skills(existing_skills, extra_skills)
    metadata["solution_summary"] = solution_summary

    # update record
    record["metadata"] = metadata

    # remove raw SQL solution from final output
    if "solution_raw" in record:
        del record["solution_raw"]

    return record

def main(limit: int = None):
    with IN_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if limit is not None:
        records = records[:limit]

    annotated_records = []
    for idx, r in enumerate(records):
        annotated = annotate_one_problem(r)
        annotated_records.append(annotated)
        print(f"processed record #{idx+1}: {annotated['metadata']['id']}")

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        json.dump(annotated_records, out_f, indent=2, ensure_ascii=False)

    print(f"\nAnnotated file saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
