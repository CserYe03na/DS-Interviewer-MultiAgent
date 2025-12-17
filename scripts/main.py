import json
from openai import OpenAI

from scripts.skill_analyzer_agent import SkillAnalyzerAgent
from scripts.scope_planner_agent import ScopePlannerAgent
from scripts.input_analyzer import TEMPLATE_A
from scripts.skill_mapper import TEMPLATE_B

def main():
    client = OpenAI()

    taxonomy_path = "data/taxonomy_skills.json"

    agent = SkillAnalyzerAgent(
        client=client,
        taxonomy_path=taxonomy_path,
        template_A=TEMPLATE_A,
        template_B=TEMPLATE_B,
    )

    jd = """
    Responsibilities:
    - Conduct data analysis in LIVE related business, including watching experience, creator ecosystem, agency management, algorithm improvement and etc..
    - Design metrics framework to measure product healthiness, keep tracking of core metrics and understand root causes of metric movements.
    - Conduct scientific evaluation with statistical methods, including A/B testing and casual inference.
    - Identify growth opportunities with data analytics, and drive business decisions. Work with PM/MLE/RD to deliver product and strategy improvement.
    - Research data science theories and methodology, improve analysis efficiency and data product tools.

    Minimum Qualifications:
    - Currently pursuing an Undergraduate/Master in Math, Statistic, Data Science, Machine Learning, or a related technical discipline.
    - Expertise in SQL and programming in Python or R.
    - Strong analytical and causal reasoning mindset, and rigidity on statistical correctness. Strong communication and passion about product challenges.

    Preferred Qualifications:
    - Experience of LIVE related business.
    - Knowledge of machine learning and recommendation systems.
    - Experience with big data technologies.
    """
    user = """
    I really want to enhance my SQL coding ability, especially with window functions and more complex analytical queries. I also need help strengthening my Pandas coding skills, random forest related machine learning knowledge as well.
    """

    
    weights = agent.run(jd_text=jd, user_desc=user)

    
    # print("\n===== Extracted Keywords =====")
    # print(json.dumps(weights["extracted"], indent=2))

    # print("\n===== Mapped Taxonomy Skills =====")
    # print(json.dumps(weights["mapped"], indent=2))

    print("\n===== Mapped Taxonomy Skills =====")
    print(json.dumps(weights["plan"], indent=2))

    scope_planner = ScopePlannerAgent(client)

    final_plan = scope_planner.run(
        user_desc=user,
        jd_text=jd,
        skill_weights=weights["plan"],
    )

    print("\n===== Final Planner Output =====")
    print(json.dumps(final_plan, indent=2))


if __name__ == "__main__":
    main()
