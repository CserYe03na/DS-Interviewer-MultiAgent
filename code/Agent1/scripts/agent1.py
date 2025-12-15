from Agent1.scripts.input_analyzer import InputAnalyzer
from Agent1.scripts.skill_mapper import SkillMapper, WeightAllocator


class SkillPlanningAgent:
    """
    Agent1: integrates Module A (input analyzer),
            Module B (skill mapper),
            Module C (weight allocator).

    API:
        agent = SkillPlanningAgent(...)
        result = agent.run(jd_text, user_desc, total_questions)
    """

    def __init__(self, client, taxonomy_path, template_A, template_B):
        """
        client: OpenAI client instance
        taxonomy_path: path to taxonomy_skills.json
        template_A: keyword extraction prompt
        template_B: mapping prompt template
        """
        self.analyzer = InputAnalyzer(client, template_A)
        self.mapper = SkillMapper(client, taxonomy_path, template_B)
        self.allocator = WeightAllocator()

    def run(self, jd_text, user_desc, total_questions):
        """
        Full pipeline execution:
        A → extract keywords
        B → map to taxonomy
        C → allocate question counts
        """
        # A: extract raw keywords
        extracted = self.analyzer.extract_keywords(jd_text, user_desc)

        # B: map natural keywords to taxonomy
        mapped = self.mapper.map(extracted)

        # C: allocate questions
        plan = self.allocator.allocate(mapped, total_questions)

        return {
            "extracted": extracted,
            "mapped": mapped,
            "plan": plan
        }
