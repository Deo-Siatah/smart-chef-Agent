"""
Lightweight eval set — NOT a full LangSmith dataset pipeline (that's
explicitly out of scope per the roadmap). This is a plain Python script
you run manually after changes, asserting on the parts of behavior that
matter most: safety-critical filtering and basic pipeline correctness.
"""
from langchain_core.messages import HumanMessage
from src.graph import graph

EVAL_CASES = [
    {
        "name": "peanut_allergy_excludes_peanut_dishes",
        "input": {
            "user_input": "I have chicken, peanuts, and rice",
            "ingredients": ["chicken", "peanuts", "rice"],
            "allergies": ["peanuts"],
            "dietary_restriction": None,
            "goal": None,
            "perishable_ingredients": [],
            "retry_count": 0,
            "candidate_recipes": [], "final_recommendation": "",
            "selected_recipe": None, "recipe_details": "", "nutrition_results": [],
        },
        "assert_fn": lambda result: all(
        "peanut" not in " ".join(r.get("used_ingredients", []) + r.get("missing_ingredients", [])).lower()
        for r in result.get("candidate_recipes", [])
        ),
        "failure_message": "A candidate recipe still contains peanut-related ingredients after allergy filtering",
    },
    {
        "name": "fish_allergy_excludes_sardines",
        "input": {
            "user_input": "I have tomatoes, sardines,onions and rice",
            "ingredients": ["tomatoes", "sardines", "onions", "rice"],
            "allergies": ["fish"],
            "dietary_restriction": None,
            "goal": None,
            "perishable_ingredients": [],
            "retry_count": 0,
            "candidate_recipes": [], "final_recommendation": "",
            "selected_recipe": None, "recipe_details": "", "nutrition_results": [],
        },
        "assert_fn": lambda result: all(
        "sardine" not in " ".join(r.get("used_ingredients", []) + r.get("missing_ingredients", [])).lower()
        for r in result.get("candidate_recipes", [])
        ),
        "failure_message": "A candidate recipe still contains sardine-related ingredients after allergy filtering",
    },
    {
        "name": "no_calorie_hallucination_when_available",
        "input": {
            "user_input": "I have kale, cassava, and beans, trying to lose weight",
            "ingredients": ["kale", "cassava", "beans"],
            "allergies": [],
            "dietary_restriction": None,
            "goal": "weight_loss",
            "perishable_ingredients": [],
            "retry_count": 0,
            "candidate_recipes": [], "final_recommendation": "",
            "selected_recipe": None, "recipe_details": "", "nutrition_results": [],
        },
        "assert_fn": lambda result: "not available" in result["final_recommendation"].lower()
                                     or "calorie" not in result["final_recommendation"].lower(),
        "failure_message": "Response may have fabricated calorie data instead of saying 'not available'",
    },
    {
        "name":"greeting_does_not_trigger_recipe_pipeline",
        "input": {
            "user_input": "hello, I'm testing this app",
            "ingredients": [], "allergies": [], "dietary_restriction": None,
            "goal": None, "perishable_ingredients": [], "retry_count": 0,
            "candidate_recipes": [], "final_recommendation": "",
            "selected_recipe": None, "recipe_details": "", "nutrition_results": [],
        
        },
        "assert_fn": lambda result: result.get("candidate_recipes") == [],
        "failure_message": "A plain greeting incorrectly triggered the recipe search pipeline",
    },
]

def run_eval():
    passed, failed = 0, 0
    for case in EVAL_CASES:
        config = {"configurable": {"thread_id": f"eval-{case['name']}", "user_id": "eval-bot"}}
        try:
            result = graph.invoke(case["input"], config=config)
            # Some cases may pause on the select_recipe interrupt — treat
            # reaching that point as "pipeline ran successfully enough to check".
            if case["assert_fn"](result):
                print(f"[PASS] {case['name']}")
                passed += 1
            else:
                print(f"[FAIL] {case['name']}: {case['failure_message']}")
                failed += 1
        except Exception as e:
            print(f"FAIL {case['name']}: {e}")
            failed += 1

    print(f"\n {passed} passed, {failed} failed out of {len(EVAL_CASES)}.")

if __name__ == "__main__":
    run_eval()