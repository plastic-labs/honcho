import dspy

gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')

class MessageResponseAssess(dspy.Signature):
    """Assess the quality of a response along the specified dimension."""
    chat_input = dspy.InputField()
    assessment_dimension = dspy.InputField()  # user state
    example_response = dspy.InputField()
    ai_response_label = dspy.OutputField(desc="yes or no")


def metric(example, pred, trace=None):
    """Assess the quality of a response along the specified dimension."""

    chat_input = example.chat_input
    assessment_dimension = f"The user is in the following state: {example.assessment_dimension}. Is the AI response appropriate for this state? Respond with Yes or No."
    example_response = pred.response

    with dspy.context(lm=gpt4T):
        assessment_result = dspy.Predict(MessageResponseAssess)(
            chat_input=chat_input, 
            assessment_dimension=assessment_dimension,
            example_response=example_response
        )

    is_appropriate = assessment_result.ai_response_label.lower() == 'yes'

    print("======== OPTIMIZER HISTORY ========")
    gpt4T.inspect_history(n=5)
    print("======== END OPTIMIZER HISTORY ========")
    
    return is_appropriate