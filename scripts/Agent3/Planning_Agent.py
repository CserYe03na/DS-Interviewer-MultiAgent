import json
import random
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

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
        meta = r.get("metadata", {}) or {}
        data = r.get("data", {}) or {}
        primary_url = meta.get("url") or meta.get("source_url")
        backup_url = data.get("backup_url")

        aligned.append({
            "score": None,
            "id": r.get("id"),
            "type": meta.get("type"),
            "title": meta.get("title"),
            "difficulty": meta.get("difficulty"),
            "taxonomy_skills": meta.get("taxonomy_skills", []),
            "url": primary_url,
            "backup_url": backup_url,
            "preview": r.get("vector_text", "")[:180].replace("\n", " "),
            "metadata": meta,
            "data": r.get("data", {})
        })
    return aligned

def normalize_tasks(raw_tasks):
    normalized = []
    for r in raw_tasks:
        meta = r.get("metadata", {}) or {}
        data = r.get("data", {}) or {}

        task_id = r.get("id")
        title = r.get("title") or meta.get("title")
        difficulty = r.get("difficulty") or meta.get("difficulty")

        if not task_id or not title or not difficulty:
            continue

        skills = r.get("taxonomy_skills") or meta.get("taxonomy_skills", []) or []
        if isinstance(skills, str):
            skills = [skills]

        normalized.append({
            "id": task_id,
            "type": r.get("type") or meta.get("type"),
            "category": meta.get("category"),
            "title": title,
            "difficulty": difficulty,
            "taxonomy_skills": skills,
            "url": r.get("url") or meta.get("url"),
            "backup_url": r.get("backup_url") or data.get("backup_url")
        })
    return normalized


# USER PARSING  find strength and weakness of user in text
def parse_request(text: str) -> int:
    m = re.search(r"(\d+)\s*days?\b", text.lower())
    return int(m.group(1)) if m else 7


def build_user_profile(text: str):
    t = text.lower()
    strong, weak = set(), set()

    if "strong in" in t:
        part = t.split("strong in", 1)[1]
        part = part.split("weak in", 1)[0]
        strong |= {s.strip() for s in part.split(",") if s.strip()}

    if "weak in" in t:
        part = t.split("weak in", 1)[1]
        part = part.split("strong in", 1)[0]
        weak |= {s.strip() for s in part.split(",") if s.strip()}

    return {"strong": strong, "weak": weak}

def build_difficulty_curve(n):
    """
    Smooth difficulty curve:
    ramp up → peak → taper for review   The difficulty distribution we want
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
# Adjust the hardness of questions based on user's familarity with that topic / skill
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



#Skipping the reviewer and summary llm, return the plans generated by logic
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

    MAX_SKILLS = 2   # Maximum number of questions should we focus on each day

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


# Using LLM reviewer to review, see any problem sequence to change
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

    # deterministic planner
    days = generate_study_plan_v2(tasks, user_request)

    summaries = None

    # V3 (LLM)
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



# MAIN (CLI)
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
            if t.get("url"):
                print(f"      ↳ {t['url']}")

        if i in summary_map:
            print("Summary:", summary_map[i])


if __name__ == "__main__":
    main()




"""
Exampel Output (with LLM)

DAY 1 — difficulty 13
  • lc:algo_1825 — Minimum Hours of Training to Win a Competition (easy)
      ↳ https://leetcode.com/problems/minimum-hours-of-training-to-win-a-competition
  • lc:algo_2218 — Find Maximum Non-decreasing Array Length (hard)
      ↳ https://leetcode.com/problems/find-maximum-non-decreasing-array-length
  • lc:algo_1379 — Count Pairs With XOR in a Range (hard)
      ↳ https://leetcode.com/problems/count-pairs-with-xor-in-a-range
  • lc:algo_1542 — The Score of Students Solving Math Expression (hard)
      ↳ https://leetcode.com/problems/the-score-of-students-solving-math-expression
  • lc:algo_382 — Smallest Good Base (hard)
      ↳ https://leetcode.com/problems/smallest-good-base
Summary: Great start with a mix of easy and hard problems! Focus on 'Minimum Hours of Training to Win a Competition' (algo_2605) to build your foundational skills in problem-solving. Tackling harder questions like 'Count Pairs With XOR in a Range' will enhance your analytical thinking, which is crucial for technical interviews.

DAY 2 — difficulty 18
  • lc:algo_94 — Binary Tree Inorder Traversal (easy)
      ↳ https://leetcode.com/problems/binary-tree-inorder-traversal
  • lc:algo_1364 — Check if Binary String Has at Most One Segment of Ones (easy)
      ↳ https://leetcode.com/problems/check-if-binary-string-has-at-most-one-segment-of-ones
  • lc:algo_369 — Validate IP Address (medium)
      ↳ https://leetcode.com/problems/validate-ip-address
  • lc:algo_2284 — Maximum Palindromes After Operations (medium)
      ↳ https://leetcode.com/problems/maximum-palindromes-after-operations
  • lc:algo_1704 — Find Players With Zero or One Losses (medium)
      ↳ https://leetcode.com/problems/find-players-with-zero-or-one-losses
  • lc:algo_888 — Longest Well-Performing Interval (medium)
      ↳ https://leetcode.com/problems/longest-well-performing-interval
  • lc:algo_2399 — Find the Minimum Area to Cover All Ones I (medium)
      ↳ https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-i
  • lc:algo_1125 — Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts (medium)
      ↳ https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts
  • lc:algo_12 — Integer to Roman (medium)
      ↳ https://leetcode.com/problems/integer-to-roman
  • lc:algo_639 — Masking Personal Information (medium)
      ↳ https://leetcode.com/problems/masking-personal-information
Summary: You're making excellent progress! Begin with 'Binary Tree Inorder Traversal' (algo_2606) to strengthen your understanding of tree traversal techniques. As you move to medium problems like 'Longest Well-Performing Interval', you'll practice critical thinking and time complexity analysis, both vital for coding interviews.

DAY 3 — difficulty 12
  • th:theory_00057 — What is feature selection? Why do we need it? (easy)
  • lc:algo_163 — Majority Element (easy)
      ↳ https://leetcode.com/problems/majority-element
  • lc:algo_108 — Convert Sorted Array to Binary Search Tree (easy)
      ↳ https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree
  • lc:algo_938 — Count Vowels Permutation (hard)
      ↳ https://leetcode.com/problems/count-vowels-permutation
  • lc:algo_800 — Subarrays with K Different Integers (hard)
      ↳ https://leetcode.com/problems/subarrays-with-k-different-integers
  • lc:algo_2647 — Shortest Path in a Weighted Tree (hard)
      ↳ https://leetcode.com/problems/shortest-path-in-a-weighted-tree
Summary: Keep up the momentum! Start with 'What is feature selection? Why do we need it?' to grasp essential concepts in data science. As you tackle 'Count Vowels Permutation', you'll enhance your combinatorial problem-solving skills, which are often tested in interviews.

DAY 4 — difficulty 12
  • lc:algo_365 — Island Perimeter (easy)
      ↳ https://leetcode.com/problems/island-perimeter
  • th:theory_00157 — What are good baselines when building a recommender system? (easy)
  • th:theory_00072 — How do we know how many trees we need in random forest? (easy)
  • lc:algo_2451 — Final Array State After K Multiplication Operations II (hard)
      ↳ https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-ii
  • lc:algo_881 — Parsing A Boolean Expression (hard)
      ↳ https://leetcode.com/problems/parsing-a-boolean-expression
  • lc:algo_557 — Parse Lisp Expression (hard)
      ↳ https://leetcode.com/problems/parse-lisp-expression
Summary: You're doing fantastic! Begin with 'Island Perimeter' to solidify your understanding of grid-based problems. As you progress to 'Final Array State After K Multiplication Operations II', you'll sharpen your skills in algorithm design and optimization, which are crucial for technical interviews.

DAY 5 — difficulty 22
  • lc:algo_2605 — Check If Digits Are Equal in String After Operations I (easy)
      ↳ https://leetcode.com/problems/check-if-digits-are-equal-in-string-after-operations-i
  • lc:algo_88 — Merge Sorted Array (easy)
      ↳ https://leetcode.com/problems/merge-sorted-array
  • lc:algo_1112 — Number of Students Doing Homework at a Given Time (easy)
      ↳ https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time
  • lc:algo_2219 — Matrix Similarity After Cyclic Shifts (easy)
      ↳ https://leetcode.com/problems/matrix-similarity-after-cyclic-shifts
  • lc:algo_2055 — Minimum String Length After Removing Substrings (easy)
      ↳ https://leetcode.com/problems/minimum-string-length-after-removing-substrings
  • lc:algo_1124 — Maximum Product of Two Elements in an Array (easy)
      ↳ https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array
  • lc:algo_622 — Binary Tree Pruning (medium)
      ↳ https://leetcode.com/problems/binary-tree-pruning
  • lc:algo_867 — Letter Tile Possibilities (medium)
      ↳ https://leetcode.com/problems/letter-tile-possibilities
  • lc:algo_404 — Find Largest Value in Each Tree Row (medium)
      ↳ https://leetcode.com/problems/find-largest-value-in-each-tree-row
  • lc:algo_1456 — Count Sub Islands (medium)
      ↳ https://leetcode.com/problems/count-sub-islands
  • lc:algo_1394 — Minimum Sideway Jumps (medium)
      ↳ https://leetcode.com/problems/minimum-sideway-jumps
  • lc:algo_2458 — K-th Nearest Obstacle Queries (medium)
      ↳ https://leetcode.com/problems/k-th-nearest-obstacle-queries
  • lc:algo_1069 — Count Number of Teams (medium)
      ↳ https://leetcode.com/problems/count-number-of-teams
  • lc:SQL_82 — Investments In 2016 (medium)
      ↳ https://leetcode.com/problems/investments-in-2016/
Summary: Great job on reaching day 5! Start with 'Check If Digits Are Equal in String After Operations I' to reinforce your string manipulation skills. Moving on to 'Binary Tree Pruning' will help you practice tree algorithms, a common topic in coding interviews.

DAY 6 — difficulty 17
  • lc:algo_1717 — Calculate Digit Sum of a String (easy)
      ↳ https://leetcode.com/problems/calculate-digit-sum-of-a-string
  • lc:algo_442 — Array Nesting (medium)
      ↳ https://leetcode.com/problems/array-nesting
  • lc:algo_989 — Sum of Mutated Array Closest to Target (medium)
      ↳ https://leetcode.com/problems/sum-of-mutated-array-closest-to-target
  • lc:algo_900 — Binary Tree Coloring Game (medium)
      ↳ https://leetcode.com/problems/binary-tree-coloring-game
  • lc:algo_405 — Longest Palindromic Subsequence (medium)
      ↳ https://leetcode.com/problems/longest-palindromic-subsequence
  • lc:algo_342 — Find All Anagrams in a String (medium)
      ↳ https://leetcode.com/problems/find-all-anagrams-in-a-string
  • lc:algo_2404 — Find the Maximum Length of Valid Subsequence II (medium)
      ↳ https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-ii
  • lc:algo_1714 — Number of Ways to Buy Pens and Pencils (medium)
      ↳ https://leetcode.com/problems/number-of-ways-to-buy-pens-and-pencils
  • lc:algo_116 — Populating Next Right Pointers in Each Node (medium)
      ↳ https://leetcode.com/problems/populating-next-right-pointers-in-each-node
Summary: You're almost there! Begin with 'Calculate Digit Sum of a String' to warm up your problem-solving skills. As you tackle 'Longest Palindromic Subsequence', you'll enhance your dynamic programming abilities, which are essential for many technical interviews.

"""