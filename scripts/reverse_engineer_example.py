"""
Example usage of the reverse_engineer module.

This script demonstrates how to use the reverse engineering functionality
to generate ground truth atomic propositions and deductions from a conversation
history, question, and correct answer.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.reverse_engineer import reverse_engineer_trace


async def main():
    # Example from LongMemEval dataset - Seco de Cordero recipe
    messages = [
        {
            "role": "user",
            "content": "Could you suggest a recipe for a classic dish from Ancash?",
        },
        {
            "role": "assistant",
            "content": """Sure! How about "Seco de Cordero" (Lamb Stew) from Ancash:

Ingredients:
- 2 lbs. lamb, cut into small pieces
- 1 onion, chopped
- 3 garlic cloves, minced
- 1 tsp. cumin
- 1 tsp. dried oregano
- 2 tbsp. vegetable oil
- 2 tbsp. ají amarillo paste
- 1 cup beer
- 2 cups water
- 4 potatoes, peeled and cut into wedges
- Salt and pepper, to taste
- 1/4 cup chopped cilantro

Instructions:
1. In a large pot, heat the oil over medium-high heat. Add the lamb pieces and cook until browned on all sides, about 8-10 minutes.
2. Remove the lamb from the pot and set it aside on a plate.
3. In the same pot, add the chopped onion and garlic. Cook until translucent, about 5-7 minutes.
4. Add the cumin, oregano, and ají amarillo paste to the pot. Stir to combine and cook for another minute or two.
5. Add the beer, water, and lamb back to the pot. Bring the mixture to a boil and then reduce the heat to low. Cover and let simmer for about 1 hour, or until the lamb is tender.
6. Add the potatoes to the pot and season with salt and pepper to taste. Cook until the potatoes are soft and the stew has thickened, about 30-45 minutes.
7. To finish, stir in the chopped cilantro and serve hot with a side of rice.

Enjoy your Seco de Cordero, a delicious and hearty dish from Ancash!""",
        },
        {
            "role": "user",
            "content": "That sounds amazing! I've never tried a lamb stew before. Do you have any recommendations for a good type of beer to use in the recipe?",
        },
        {
            "role": "assistant",
            "content": "I don't have personal preferences, but in general, it's recommended to use a light or medium-bodied beer for this recipe to avoid overpowering the flavors of the lamb and spices. A Pilsner or Lager would work well, but you can also use a local beer from Ancash to give the dish a more authentic touch. Ultimately, it comes down to your personal preference, so you can choose a beer that you enjoy drinking and experiment with different types to find your favorite one. Enjoy your delicious Seco de Cordero!",
        },
    ]

    question = "I'm looking back at our previous conversation about the Seco de Cordero recipe from Ancash. You mentioned using a light or medium-bodied beer, but I was wondering if you could remind me what type of beer you specifically recommended?"

    answer = "I recommended using a Pilsner or Lager for the recipe."

    print("=" * 80)
    print("REVERSE ENGINEERING EXAMPLE")
    print("=" * 80)
    print("\nQuestion:")
    print(question)
    print("\nCorrect Answer:")
    print(answer)
    print("\nGenerating minimal set of atomic propositions and deductions...")
    print("=" * 80)

    # Call the reverse engineering function (single unified LLM call)
    (
        explicit_response,
        deductive_response,
        observer_card,
        observed_card,
    ) = await reverse_engineer_trace(
        messages=messages,
        question=question,
        answer=answer,
        observer="user",
        observed="assistant",
    )

    print("\n--- EXPLICIT OBSERVATIONS ---")
    if explicit_response.explicit:
        for i, obs in enumerate(explicit_response.explicit, 1):
            print(f"{i}. {obs.content}")
    else:
        print("(none)")

    print("\n--- IMPLICIT OBSERVATIONS ---")
    if explicit_response.implicit:
        for i, obs in enumerate(explicit_response.implicit, 1):
            print(f"{i}. {obs.content}")
    else:
        print("(none)")

    print("\n--- DEDUCTIVE OBSERVATIONS ---")
    if deductive_response.deductions:
        for i, obs in enumerate(deductive_response.deductions, 1):
            print(f"{i}. Conclusion: {obs.conclusion}")
            if obs.premises:
                print("   Premises:")
                for premise in obs.premises:
                    print(f"   - {premise}")
    else:
        print("(none)")

    print("\n--- OBSERVER PEER CARD ---")
    if observer_card:
        for i, card_entry in enumerate(observer_card, 1):
            print(f"{i}. {card_entry}")
    else:
        print("(none)")

    print("\n--- OBSERVED PEER CARD ---")
    if observed_card:
        for i, card_entry in enumerate(observed_card, 1):
            print(f"{i}. {card_entry}")
    else:
        print("(none)")

    print("\n" + "=" * 80)
    print("JSON OUTPUT")
    print("=" * 80)

    output = {
        "explicit": [obs.content for obs in explicit_response.explicit],
        "implicit": [obs.content for obs in explicit_response.implicit],
        "deductions": [
            {"premises": obs.premises, "conclusion": obs.conclusion}
            for obs in deductive_response.deductions
        ],
        "observer_card": observer_card,
        "observed_card": observed_card,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
