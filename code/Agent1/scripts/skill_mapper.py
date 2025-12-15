import json
from string import Template
from typing import List
from openai import OpenAI


def load_taxonomy(path: str) -> List[str]:
    with open(path, "r") as f:
        taxonomy = json.load(f)
    if not isinstance(taxonomy, list):
        raise ValueError("taxonomy_skills.json must contain a list of strings.")
    return taxonomy


SQL_BASE_PACK = [
    "sql_easy", 
    "sql_basic", 
    "sql_aggregation",
    "sql_filter", 
    "sql_group_by", 
    "sql_sorting"
]

PANDAS_BASE_PACK = [
    "pandas_easy",
    "algo_easy",
    "pandas_data_cleaning",
    "pandas_data_selection",
    "pandas_data_reshaping",
    "pandas_data_manipulation",
    "pandas_data_inspection"
]

ALGO_BASE_PACK = [
    "algo_easy", 
    "algo_array_string",
    "algo_hash_map",
    "algo_sorting",
    "algo_dp"
]


TEMPLATE_B = Template("""
You are a skill-mapping module in a multi-agent hiring assistant.

Your job:
Map one natural-language skill phrase to taxonomy skills.

You will receive:
- skill_phrase: "$KEYWORD"
- taxonomy: $TAXONOMY

RULES:

1. Match the phrase to the most appropriate taxonomy skills.
2. You may return multiple skills if relevant.
3. If no taxonomy skill matches, return an empty list.
4. DO NOT invent skills — only return items from taxonomy.
5. Use semantic similarity, common usage, and DS interview knowledge.
6. Use the following base skill packs only as hints:

SQL base pack:
$SQL_BASE_PACK

Pandas base pack:
$PANDAS_BASE_PACK

Algorithms base pack:
$ALGO_BASE_PACK

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON, no markdown fences, no comments.

{
  "mapped_skills": []
}
""")


class SkillMapper:
    """
    Module B: Convert extracted natural-language keywords to taxonomy skills
    using the LLM mapping prompt (TEMPLATE_B).
    """

    def __init__(self, client: OpenAI, taxonomy_path: str, template=TEMPLATE_B):
        self.client = client
        self.template = template
        self.taxonomy = load_taxonomy(taxonomy_path)

    # ---- LLM single-skill mapping ----
    def _map_single(self, keyword: str) -> List[str]:
        prompt = self.template.substitute(
            KEYWORD=keyword,
            TAXONOMY=json.dumps(self.taxonomy),
            SQL_BASE_PACK=json.dumps(SQL_BASE_PACK),
            PANDAS_BASE_PACK=json.dumps(PANDAS_BASE_PACK),
            ALGO_BASE_PACK=json.dumps(ALGO_BASE_PACK),
        )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            content = resp.choices[0].message.content
            parsed = json.loads(content)
            mapped = parsed.get("mapped_skills", [])

            # Only keep valid taxonomy skills
            return [s for s in mapped if s in self.taxonomy]

        except Exception as e:
            print("LLM mapping error:", e)
            return []

    # ---- Batch mapping over all categories ----
    def map(self, extracted_keywords: dict) -> dict:
        """
        Input structure:
        {
            "required_keywords": [...],
            "preferred_keywords": [...],
            "general_keywords": [...]
        }

        Output structure:
        {
            "required_mapped_skills": { keyword: [taxonomy skills], ... },
            "preferred_mapped_skills": { ... },
            "general_mapped_skills": { ... }
        }
        """

        output = {}

        for category, kw_list in extracted_keywords.items():
            mapped_category = category.replace("keywords", "mapped_skills")
            output[mapped_category] = {}

            for kw in kw_list:
                mapped_skills = self._map_single(kw)
                output[mapped_category][kw] = mapped_skills

        return output


class WeightAllocator:
    """
    Module C: Allocate study question counts based on mapped taxonomy skills
    and user-selected total question count.
    """

    def __init__(self):
        # default baseline weights
        self.base_weights = {
            "required_mapped_skills": 3.0,
            "preferred_mapped_skills": 2.0,
            "general_mapped_skills": 1.0,
        }

        # minimum questions per category (priority forcing)
        self.min_questions = {
            "required_mapped_skills": 5,
            "preferred_mapped_skills": 3,
            "general_mapped_skills": 1,
        }

        self.repeat_bonus = 0.5

    def allocate(self, mapped_skills: dict, total_questions: int) -> dict:
        """
        mapped_skills example:
        {
            "required_mapped_skills": {"SQL": ["sql_basic"], ...},
            "preferred_mapped_skills": {...},
            "general_mapped_skills": {...}
        }
        """

        skill_info = {}

        for group, kw_dict in mapped_skills.items():
            for phrase, skills in kw_dict.items():
                for s in skills:
                    if s not in skill_info:
                        skill_info[s] = {"groups": set(), "count": 0}
                    skill_info[s]["groups"].add(group)
                    skill_info[s]["count"] += 1

        if not skill_info:
            return {}

        final_weights = {}

        for skill, info in skill_info.items():
            groups = info["groups"]
            count = info["count"]

            baseline = max(self.base_weights[g] for g in groups)
            weight = baseline + (count - 1) * self.repeat_bonus
            final_weights[skill] = weight

        group_to_skills = {
            group: {s for kw_dict in mapped_skills.values() for kw, skills in kw_dict.items() if group in mapped_skills and s in skills}
            for group in ['required_mapped_skills','preferred_mapped_skills','general_mapped_skills']
        }

        allocated = {s: 0 for s in final_weights}
        forced_total = 0

        for group, min_q in self.min_questions.items():
            skills = group_to_skills.get(group, set())
            if not skills:
                continue

            per_skill = max(min_q // len(skills), 1)

            for s in skills:
                allocated[s] += per_skill

            forced_total += len(skills) * per_skill

        remaining = max(total_questions - forced_total, 0)

        weight_sum = sum(final_weights.values())

        for skill, weight in final_weights.items():
            allocated[skill] += round((weight / weight_sum) * remaining)

        drift = total_questions - sum(allocated.values())

        if drift != 0:
            ordered = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            while drift != 0:
                s = ordered[idx % len(ordered)][0]
                allocated[s] += 1 if drift > 0 else -1
                drift += -1 if drift > 0 else 1
                idx += 1

        return allocated

if __name__ == "__main__":
    client = OpenAI()

    extracted_keywords = {
        "required_keywords": ["SQL", "window functions"],
        "preferred_keywords": ["Pandas coding skills"],
        "general_keywords": ["Math"]
    }

    mapper = SkillMapper(client, "data/taxonomy_skills.json")
    mapped = mapper.map(extracted_keywords)

    allocator = WeightAllocator()
    plan = allocator.allocate(mapped, total_questions=40)

    print(json.dumps(mapped, indent=2))
    print(json.dumps(plan, indent=2))
