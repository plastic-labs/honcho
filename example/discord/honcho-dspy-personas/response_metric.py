import dspy

gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')

class MessageResponseAssess(dspy.Signature):
    """Assess the quality of a response along the specified dimension."""
    chat_input = dspy.InputField()
    ai_response = dspy.InputField()
    gold_response = dspy.InputField()
    assessment_dimension = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Good or not")


def metric(example, ai_response, trace=None):
    """Assess the quality of a response along the specified dimension."""

    assessment_dimension = example.assessment_dimension
    chat_input = example.chat_input
    gold_response = example.response

    with dspy.context(lm=gpt4T):
        assessment_result = dspy.Predict(MessageResponseAssess)(
            chat_input=chat_input, 
            ai_response=ai_response, 
            gold_response=gold_response,
            assessment_dimension=assessment_dimension
        )
    
    is_positive = assessment_result.assessment_answer.lower() == 'good'

    gpt4T.inspect_history(n=3)
    
    return is_positive
