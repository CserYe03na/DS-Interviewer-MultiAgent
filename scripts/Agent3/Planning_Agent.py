


"""
This module implements a Planning Agent that:
1) Takes a list of normalized tasks (from Retrieval Agent or local JSONL)
2) Builds a multi-day study plan deterministically (V2)
3) Optionally refines the plan and generates summaries using an LLM (V3)




Local testing:  包含随机抽取的 questions 和fake input
1) Create key.env:
   OPENAI_API_KEY=your_key_here

2) Run without LLM:    
   python Planning_Agent.py --no-llm  （在 terminal输入）

3) Run with LLM reviewer + summaries:
   python Planning_Agent.py

   

当在别的file调用 Planning Agent的时候

Call THIS function (ignore main()):

    run_planning_agent(
        tasks: List[Dict],
        user_request: str,
        use_llm: bool,
        client: Optional[OpenAI]
    )

Example:

    client = get_openai_client()
    days, summaries = run_planning_agent(
        tasks=normalized_tasks,
        user_request="Create a 7 day plan. Weak in ML.",
        use_llm=True,
        client=client
    )

If use_llm=False:
- pass client=None
- summaries will be None

--------------------------------------------------
INPUT FORMAT (TASK)
--------------------------------------------------
{
  "id": str,
  "title": str,
  "difficulty": "easy|medium|hard",
  "taxonomy_skills": List[str]
}

Output:
- days: List[List[task]]
- summaries: Optional[List[{day, summary}]]
"""


import json
import random
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI





#   ENV / CLIENT  set up
load_dotenv("key.env")

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


#---------------------------------Faking retrival Agent's output-------------
# DATA LOADING    读取本地的 jsonl
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def align_to_retrieval_format(raw_tasks):
    aligned = []
    for r in raw_tasks:
        meta = r.get("metadata", {})
        aligned.append({
            "score": None,
            "id": r.get("id"),
            "type": meta.get("type"),
            "title": meta.get("title"),
            "difficulty": meta.get("difficulty"),
            "taxonomy_skills": meta.get("taxonomy_skills", []),
            "preview": r.get("vector_text", "")[:180].replace("\n", " "),
            "metadata": meta,
            "data": r.get("data", {})
        })
    return aligned

#---------------------------Fetch info we need-------------
def normalize_tasks(raw_tasks):
    normalized = []
    for r in raw_tasks:
        meta = r.get("metadata", {})
        if not r.get("id") or not meta.get("title") or not meta.get("difficulty"):
            continue

        skills = meta.get("taxonomy_skills", [])
        if isinstance(skills, str):
            skills = [skills]

        normalized.append({
            "id": r["id"],
            "type": meta.get("type"),
            "category": meta.get("category"),
            "title": meta["title"],
            "difficulty": meta["difficulty"],
            "taxonomy_skills": skills
        })
    return normalized


# USER PARSING  在user input里找 总共时长以及他的强/弱项
def parse_request(text: str) -> int:
    m = re.search(r"(\d+)\s*day", text.lower())
    return int(m.group(1)) if m else 7


def build_user_profile(text: str):
    text = text.lower()
    strong, weak = set(), set()

    if "strong in" in text:
        strong |= {s.strip() for s in text.split("strong in")[1].split(",")}

    if "weak in" in text:
        weak |= {s.strip() for s in text.split("weak in")[1].split(",")}

    return {"strong": strong, "weak": weak}


# ------------------------------DIFFICULTY LOGIC--------------
def build_difficulty_curve(n):
    """
    Smooth difficulty curve:
    ramp up → peak → taper for review  我们想要的难度曲线
    """
    if n <= 0:
        return []

    raw = []
    for i in range(n):
        x = 0.0 if n == 1 else i / (n - 1)
        if x < 0.5:
            val = 0.6 + (x / 0.5) * 0.6
        else:
            val = 1.2 - ((x - 0.5) / 0.5) * 0.4
        raw.append(val)

    total = sum(raw)
    return [v / total for v in raw]

# 根据用户擅长与否调整题目默认的难度
def adjusted_difficulty(task, profile):
    base = {"easy": 1, "medium": 2, "hard": 3}[task["difficulty"]]
    skills = [s.lower() for s in task.get("taxonomy_skills", [])]
    strong = [s.lower() for s in profile["strong"]]
    weak = [s.lower() for s in profile["weak"]]

    if any(w in skill for skill in skills for w in weak):
        return base * 1.3
    if any(st in skill for skill in skills for st in strong):
        return base * 0.7
    return base

def rebalance_underfilled_days(days, day_scores, targets):
    """
    Move tasks from overloaded days to underfilled days.
    """
    num_days = len(days)

    for _ in range(2):  # two passes, same as Colab
        for i in range(num_days):
            if day_scores[i] >= targets[i]:
                continue

            # find a donor day
            donor = max(
                range(num_days),
                key=lambda j: day_scores[j] - targets[j]
            )

            if day_scores[donor] <= targets[donor]:
                continue

            # move the hardest task
            hardest = max(
                days[donor],
                key=lambda t: {"easy": 1, "medium": 2, "hard": 3}[t["difficulty"]],
                default=None
            )

            if hardest:
                days[donor].remove(hardest)
                days[i].append(hardest)

                diff = {"easy": 1, "medium": 2, "hard": 3}[hardest["difficulty"]]
                day_scores[donor] -= diff
                day_scores[i] += diff

    return days, day_scores

def cap_overfilled_days(days, day_scores, targets):
    """
    Prevent any day from exceeding 1.3x its target difficulty.
    """
    num_days = len(days)

    for i in range(num_days):
        cap = 1.3 * targets[i]

        while day_scores[i] > cap and len(days[i]) > 1:
            # move easiest task
            easiest = min(
                days[i],
                key=lambda t: {"easy": 1, "medium": 2, "hard": 3}[t["difficulty"]]
            )

            # find best recipient
            recipient = min(
                range(num_days),
                key=lambda j: day_scores[j]
            )

            days[i].remove(easiest)
            days[recipient].append(easiest)

            diff = {"easy": 1, "medium": 2, "hard": 3}[easiest["difficulty"]]
            day_scores[i] -= diff
            day_scores[recipient] += diff

    return days, day_scores



# V2 — DETERMINISTIC PLANNER --- 不使用 后续 reviewer llm 以及 smmary llm 直接返回按照逻辑的初版 plan
def assign_tasks(tasks, targets, profile):
    scored = [
        {**t, "score": adjusted_difficulty(t, profile)}
        for t in tasks
    ]
    scored.sort(key=lambda x: -x["score"])

    num_days = len(targets)
    days = [[] for _ in range(num_days)]
    day_scores = [0.0] * num_days
    day_skills = [set() for _ in range(num_days)]

    MAX_TASKS = 6
    MAX_SKILLS = 2

    # seed one task per day
    for i in range(num_days):
        if not scored:
            break
        t = scored.pop(0)
        days[i].append(t)
        day_scores[i] += t["score"]
        if t["taxonomy_skills"]:
            day_skills[i].add(t["taxonomy_skills"][0])

    for task in scored:
        task_skill = task["taxonomy_skills"][0] if task["taxonomy_skills"] else "general"
        best_day, best_cost = None, float("inf")

        for i in range(num_days):
            if len(days[i]) >= MAX_TASKS:
                continue

            skill_penalty = (
                3.0
                if task_skill not in day_skills[i]
                and len(day_skills[i]) >= MAX_SKILLS
                else 0.0
            )

            overload_penalty = max(0, day_scores[i] - targets[i])
            review_bonus = -1.0 if i >= num_days - 2 else 0.0

            cost = (
                abs((day_scores[i] + task["score"]) - targets[i])
                + skill_penalty
                + overload_penalty
                + review_bonus
            )

            if cost < best_cost:
                best_cost, best_day = cost, i

        if best_day is None:
            best_day = min(range(num_days), key=lambda i: len(days[i]))

        days[best_day].append(task)
        day_scores[best_day] += task["score"]
        day_skills[best_day].add(task_skill)

    return days, day_scores


def reorder(days):
    order = {"easy": 1, "medium": 2, "hard": 3}
    return [sorted(day, key=lambda t: order[t["difficulty"]]) for day in days]


def generate_study_plan_v2(tasks, user_text):
    days_n = parse_request(user_text)
    profile = build_user_profile(user_text)

    personalized = [adjusted_difficulty(t, profile) for t in tasks]
    total = sum(personalized)
    targets = [total * w for w in build_difficulty_curve(days_n)]

    days, day_scores = assign_tasks(tasks, targets, profile)
    days, day_scores = rebalance_underfilled_days(days, day_scores, targets)
    days, day_scores = cap_overfilled_days(days, day_scores, targets)

    return reorder(days)


# V3 — LLM REVIEWER + SUMMARY  两个 LLM reviewer review 初版assignment, see any problem sequence to change
def safe_parse_json(text):
    """
    Safely parse JSON from LLM output.
    Returns None if parsing fails.
    """
    if not text or not text.strip():
        print("[LLM ERROR] Empty response")
        return None

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # First attempt: parse directly
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second attempt: extract first JSON object or array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            print("[LLM ERROR] JSON extraction failed")
            print("RAW:", repr(text))
            print("EXTRACTED:", match.group(1))
            return None

    print("[LLM ERROR] No JSON found")
    print("RAW OUTPUT:", repr(text))
    return None


def normalize_summaries(summaries):
    """
    Ensure summaries is always:
    List[{"day": int, "summary": str}]
    """
    if summaries is None:
        return None

    # Correct case
    if isinstance(summaries, list):
        if summaries and isinstance(summaries[0], dict):
            return summaries

    # Single object → wrap
    if isinstance(summaries, dict):
        if "day" in summaries and "summary" in summaries:
            return [summaries]

    # Anything else → reject
    print("[INFO] Invalid summaries format, skipping summaries")
    return None



def build_alignment_prompt(plan):
    return f"""
You are reviewing a study plan generated by a deterministic algorithm.

Your role:
- ONLY suggest small task swaps if they improve cognitive flow.
- Do NOT add or remove tasks.
- Do NOT reorder tasks within a day.
- Prefer moving HARD tasks away from overloaded days.
- If plan looks good, return an empty swap list.

Return JSON ONLY.

Input plan:
{json.dumps(plan, indent=2)}

Output format:
{{
  "swap": [
    {{
      "from_day": 2,
      "to_day": 3,
      "task_id": "lc:algo_881"
    }}
  ],
  "notes": "optional explanation"
}}
"""

def review_plan_with_llm(days, client):
    compact = [
        {
            "day": i + 1,
            "tasks": [
                {
                    "id": t["id"],
                    "difficulty": t["difficulty"],
                    "skills": t["taxonomy_skills"]
                }
                for t in day
            ],
        }
        for i, day in enumerate(days)
    ]

    prompt = build_alignment_prompt(compact)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return safe_parse_json(response.choices[0].message.content)


def apply_llm_swaps(days, result):
    if not result or "swap" not in result:
        return days

    id_map = {t["id"]: t for d in days for t in d}

    for s in result["swap"]:
        fd, td = s["from_day"] - 1, s["to_day"] - 1
        task = id_map.get(s["task_id"])
        if task and task in days[fd]:
            days[fd].remove(task)
            days[td].append(task)

    return days


def normalize_review(review):
    if not isinstance(review, dict):
        return None
    if "swap" not in review or not isinstance(review["swap"], list):
        return None
    return review




def build_summary_prompt(compact_plan):
    return f"""
You are a supportive study coach helping with interview prep.

Rules:
- Do NOT change tasks or days.
- Write 2–3 encouraging, actionable sentences per day.
- You MAY reference question IDs (e.g., algo_2605) to help users locate problems.
- Explain what skills the user is practicing and why it matters.
- Return ONLY valid JSON. No text outside JSON.

Input:
{json.dumps(compact_plan, indent=2)}

Output format:
[
  {{
    "day": 1,
    "summary": "..."
  }}
]
"""
def summarize_days(days, client):
    compact = [
        {
            "day": i + 1,
            "tasks": [
                {"title": t["title"], "difficulty": t["difficulty"]}
                for t in day
            ],
        }
        for i, day in enumerate(days)
    ]

    prompt = build_summary_prompt(compact)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return safe_parse_json(response.choices[0].message.content)





def run_planning_agent(
    tasks: List[Dict[str, Any]],
    user_request: str,
    use_llm: bool = True,
    client: Optional[OpenAI] = None,
):
    """
    Unified entry point for Planning Agent (for orchestration).

    Parameters
    ----------
    tasks : List[Dict]
        Normalized tasks from Retrieval Agent.
    user_request : str
        Natural language user request.
    use_llm : bool
        Whether to use LLM reviewer + summary.
    client : OpenAI | None
        Required if use_llm=True.

    Returns
    -------
    days : List[List[Dict]]
        Study plan grouped by day.
    summaries : Optional[List[Dict]]
        Day-level summaries.
    """

    # ---- V2 deterministic planner ----
    days = generate_study_plan_v2(tasks, user_request)

    summaries = None

    # ---- V3 (LLM) ----
    if use_llm:
        if client is None:
            # fallback to local key.env
            client = get_openai_client()

        review = normalize_review(review_plan_with_llm(days, client))
        if review:
            days = apply_llm_swaps(days, review)

        raw_summaries = summarize_days(days, client)
        summaries = normalize_summaries(raw_summaries)

    return days, summaries




# MAIN (CLI)   包含default, fake inputs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/merged.jsonl")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--request",
        default="Create a 6 day plan. I am strong in SQL, weak in ML."
    )
    args = parser.parse_args()

    random.seed(42)

    # ---------- Load + sample data ----------
    raw = load_jsonl(Path(args.input))
    sampled = random.sample(raw, min(args.sample, len(raw)))

    # ---------- Build tasks ----------
    tasks = normalize_tasks(align_to_retrieval_format(sampled))

    # ---------- V2 planner ----------
    days = generate_study_plan_v2(tasks, args.request)

    # ---------- Optional V3 (LLM) ----------
    client = None
    if not args.no_llm:
        client = get_openai_client()

    days, summaries = run_planning_agent(
        tasks=tasks,
        user_request=args.request,
        use_llm=not args.no_llm,
        client=client
    )


    # ---------- Output ----------
    diff = {"easy": 1, "medium": 2, "hard": 3}

    summary_map = {}
    if summaries:
        summary_map = {s["day"]: s["summary"] for s in summaries}

    for i, day in enumerate(days, 1):
        print(f"\nDAY {i} — difficulty {sum(diff[t['difficulty']] for t in day)}")
        for t in day:
            print(f"  • {t['id']} — {t['title']} ({t['difficulty']})")

        if i in summary_map:
            print("Summary:", summary_map[i])

if __name__ == "__main__":
    main()




"""
Exampel Output (with LLM)

DAY 1 — difficulty 11
  • lc:algo_1112 — Number of Students Doing Homework at a Given Time (easy)
  • th:theory_00072 — How do we know how many trees we need in random forest? (easy)
  • lc:algo_365 — Island Perimeter (easy)
  • lc:algo_888 — Longest Well-Performing Interval (medium)
  • lc:algo_1542 — The Score of Students Solving Math Expression (hard)
  • lc:algo_382 — Smallest Good Base (hard)
Summary: Great start! Today, focus on understanding the basics of data structures and algorithms with the easy tasks. For example, solving 'Island Perimeter' (algo_463) will help you practice spatial reasoning, which is crucial for many coding interviews. As you tackle the harder problems like 'The Score of Students Solving Math Expression' (algo_1206), remember that persistence is key—keep refining your approach!

DAY 2 — difficulty 13
  • lc:algo_2219 — Matrix Similarity After Cyclic Shifts (easy)
  • lc:algo_1825 — Minimum Hours of Training to Win a Competition (easy)
  • lc:algo_163 — Majority Element (easy)
  • lc:algo_2404 — Find the Maximum Length of Valid Subsequence II (medium)
  • lc:algo_1714 — Number of Ways to Buy Pens and Pencils (medium)
  • lc:algo_116 — Populating Next Right Pointers in Each Node (medium)
  • lc:algo_369 — Validate IP Address (medium)
  • lc:algo_2284 — Maximum Palindromes After Operations (medium)
Summary: You're doing fantastic! Today's tasks will enhance your problem-solving skills, especially with arrays and matrices. The 'Majority Element' (algo_169) is a great way to practice counting techniques, while 'Populating Next Right Pointers in Each Node' (algo_116) will strengthen your understanding of tree structures. Keep pushing through the medium challenges—they're designed to stretch your abilities!

DAY 3 — difficulty 15
  • lc:algo_108 — Convert Sorted Array to Binary Search Tree (easy)
  • th:theory_00057 — What is feature selection? Why do we need it? (easy)
  • lc:algo_622 — Binary Tree Pruning (medium)
  • lc:algo_867 — Letter Tile Possibilities (medium)
  • lc:algo_404 — Find Largest Value in Each Tree Row (medium)
  • lc:algo_1456 — Count Sub Islands (medium)
  • lc:algo_1394 — Minimum Sideway Jumps (medium)
  • lc:algo_557 — Parse Lisp Expression (hard)
Summary: Keep up the momentum! Today’s focus on binary trees and feature selection will deepen your understanding of data structures and their applications. 'Binary Tree Pruning' (algo_669) will help you practice recursion, which is a vital skill in interviews. As you approach the harder problems, remember that tackling complex issues builds your confidence and expertise!

DAY 4 — difficulty 22
  • lc:algo_2605 — Check If Digits Are Equal in String After Operations I (easy)
  • lc:algo_88 — Merge Sorted Array (easy)
  • lc:algo_2055 — Minimum String Length After Removing Substrings (easy)
  • lc:algo_1717 — Calculate Digit Sum of a String (easy)
  • lc:algo_2458 — K-th Nearest Obstacle Queries (medium)
  • lc:algo_1069 — Count Number of Teams (medium)
  • lc:SQL_82 — Investments In 2016 (medium)
  • lc:algo_2451 — Final Array State After K Multiplication Operations II (hard)
  • lc:algo_2218 — Find Maximum Non-decreasing Array Length (hard)
  • lc:algo_938 — Count Vowels Permutation (hard)
  • lc:algo_881 — Parsing A Boolean Expression (hard)
Summary: You're making great progress! Today's tasks will sharpen your skills in string manipulation and array handling. 'Merge Sorted Array' (algo_88) is a classic problem that will enhance your understanding of sorting algorithms. As you work on the harder tasks, like 'Count Vowels Permutation' (algo_1220), challenge yourself to think outside the box—this will prepare you for unexpected questions in interviews!

DAY 5 — difficulty 18
  • lc:algo_94 — Binary Tree Inorder Traversal (easy)
  • lc:algo_1124 — Maximum Product of Two Elements in an Array (easy)
  • lc:algo_1704 — Find Players With Zero or One Losses (medium)
  • lc:algo_2399 — Find the Minimum Area to Cover All Ones I (medium)
  • lc:algo_1125 — Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts (medium)
  • lc:algo_12 — Integer to Roman (medium)
  • lc:algo_639 — Masking Personal Information (medium)
  • lc:algo_800 — Subarrays with K Different Integers (hard)
  • lc:algo_1379 — Count Pairs With XOR in a Range (hard)
Summary: Fantastic effort! Today, you'll work on tree traversal and array manipulation, which are key areas in coding interviews. 'Binary Tree Inorder Traversal' (algo_94) will solidify your understanding of tree structures, while 'Maximum Product of Two Elements in an Array' (algo_1464) will help you practice optimization techniques. Embrace the challenges of the harder problems—they're stepping stones to mastering coding interviews!

DAY 6 — difficulty 15
  • th:theory_00157 — What are good baselines when building a recommender system? (easy)
  • lc:algo_1364 — Check if Binary String Has at Most One Segment of Ones (easy)
  • lc:algo_442 — Array Nesting (medium)
  • lc:algo_989 — Sum of Mutated Array Closest to Target (medium)
  • lc:algo_900 — Binary Tree Coloring Game (medium)
  • lc:algo_405 — Longest Palindromic Subsequence (medium)
  • lc:algo_342 — Find All Anagrams in a String (medium)
  • lc:algo_2647 — Shortest Path in a Weighted Tree (hard)
Summary: You're almost there! Today's tasks will enhance your skills in array operations and string handling. 'Array Nesting' (algo_565) is a great way to practice working with indices and loops. As you tackle the harder problem, 'Shortest Path in a Weighted Tree' (algo_1631), remember that understanding graph theory can set you apart in technical interviews. Keep pushing forward—you’re doing amazing!
"""