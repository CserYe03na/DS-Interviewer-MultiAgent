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
Planning_Agent.py       # Generates a study plan by allocating retrieved tasks across days using difficulty curves, skill balance, and LLM-based refinement and summaries
orchestration.py        # Orchestrates agent workflows and manages state
