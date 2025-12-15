import json
from openai import OpenAI

class InputAnalyzer:
    """
    Module A: Extract raw skill phrases from JD + User description.
    This class encapsulates keyword extraction logic so that
    it can be easily used inside a multi-agent system.
    """

    def __init__(self, client: OpenAI, template: str):
        """
        client: OpenAI client instance
        template: the extraction prompt template
        """
        self.client = client
        self.template = template

    def extract_keywords(self, jd_text: str = "", user_desc: str = "") -> dict:
        """
        Extracts required_keywords, preferred_keywords, general_keywords
        based on JD + user description.

        At least one of jd_text or user_desc must be non-empty.
        """

        if not jd_text and not user_desc:
            raise ValueError("Both jd_text and user_desc are empty. Provide at least one input.")

        # Construct the prompt
        prompt = self.template

        if jd_text.strip():
            prompt += "\n\n### Job Description:\n" + jd_text.strip()

        if user_desc.strip():
            prompt += "\n\n### User Description:\n" + user_desc.strip()

        prompt += "\n\n### JSON Output:\n"

        # LLM call
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract and categorize skill keywords with maximum accuracy."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )

        raw = response.choices[0].message.content

        # JSON parsing
        try:
            data = json.loads(raw)
        except:
            try:
                cleaned = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(cleaned)
            except Exception as e:
                print("[ERROR] Could not parse model output as JSON:\n", raw)
                raise e

        return data

## Module A: Keyword & Time limits extractor from inputs

TEMPLATE_A = """
You are a skill extraction module inside a multi-agent hiring assistant.
Your goal is to extract **technical skill** phrases from two sources:

1. Job Description (JD)
2. User's self-description of what they want to review for a data-related interview

and categorize the extracted skill phrases into required, preferred, and general.

-----
CLASSIFICATION RULES
-----

A. REQUIRED KEYWORDS ("required_keywords")
Assign a skill to this category ONLY IF:

1. It appears in the JD with an explicitly mandatory tone:
    - Strong mandatory wording:
        * "must have", "required", "need to", "expected to",
          "minimum qualification", "basic qualification", "strong experience",
          "expert in", "proficient in", "deep knowledge",
          "demonstrated ability", "rigorous knowledge"

2. OR it appears in the user's description with **strongly urgent or emphatic intent**, such as:
    - "I must get better at..."
    - "I urgently need to strengthen..."
    - "I struggle with X and must learn it"
    - "I absolutely need to master..."

IMPORTANT:
1. Phrase patterns such as:
    - "I need help strengthening..."
    - "I need help with..."
express a request for support, NOT urgency.  
Skills in these cases should NOT be classified as required unless emphasis rule applies.

2. The word "also" or similar connection words MUST NOT be interpreted as an extension of the previous sentence's intensity.
    “also” is conversational filler and does NOT:
    - increase priority
    - strengthen user intent
    - indicate strong necessity
    - extend the emphasis scope of words like “especially”, “particularly”, etc.

    If a skill appears in a sentence starting with or containing “also”, 
    it MUST be treated as a normal user-requested skill by default 
    and should be placed under preferred_keywords unless it contains separate strong urgency wording 
    (e.g., “I absolutely need to master…”) or contains its OWN emphasis indicators 
    (e.g., “especially X”, “specifically Y”).


3. OR — **IMPORTANT RULE** —
   The skill appears immediately after an **emphasis indicator**, which signals priority focus.
   These indicators include:
       "really", "especially", "particularly", "in particular", 
       "specifically", "mainly", "mostly"

   Any skill phrase that follows an emphasis indicator MUST be classified as required, even if the overall sentence is not urgent.

   Example:
       "I want to improve SQL, especially window functions and complex analytical queries."
       → "SQL" = preferred
       → "window functions" and "complex analytical queries" = required

B. PREFERRED KEYWORDS ("preferred_keywords")
Assign a skill here IF:

1. It appears in the JD with soft preference tone:
    - Soft words: "preferred", "nice to have", "bonus", "optional",
                  "helpful", "good to have", "plus", "preferred qualification"

2. OR it appears in the user's description as a desired skill to learn or review,  

Examples:
    - "I need to improve..."
    - "I want to learn..."
    - "I'd like to review..."
    - "I hope to strengthen..."
    - "I want more practice on..."


C. GENERAL KEYWORDS ("general_keywords")
Assign a skill here IF:

- It appears in the JD but with neutral tone and not explicitly required/preferred.
- Or it represents domain knowledge, tasks, or methods mentioned indirectly.
- Or it is extracted as a technical concept but not clearly tied to requirement or user need.

-----
OUTPUT FORMAT
-----
Return ONLY valid JSON in this exact structure:

{
    "required_keywords": [...],
    "preferred_keywords": [...],
    "general_keywords": [...]
}

Do NOT map to a taxonomy.
Extract only raw natural-language skill phrases.
No explanations, no comments.
"""


## Example usage
def main():

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
    client = OpenAI()
    analyzer = InputAnalyzer(client=client, template=TEMPLATE_A)
    extracted = analyzer.extract_keywords(jd_text=jd, user_desc=user)
    print(json.dumps(extracted, indent=2))

if __name__ == "__main__":
    main()