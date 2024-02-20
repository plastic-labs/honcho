import dspy

gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')

class MessageResponseAssess(dspy.Signature):
    """Assess the quality of a response along the specified dimension."""
    user_message = dspy.InputField()
    ai_response = dspy.InputField()
    assessment_dimension = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Good or not")


def assess_response_quality(user_message, ai_response, assessment_dimension):
    with dspy.context(lm=gpt4T):
        assessment_result = dspy.Predict(MessageResponseAssess)(
            user_message=user_message, 
            ai_response=ai_response, 
            assessment_dimension=assessment_dimension
        )
    
    is_positive = assessment_result.assessment_answer.lower() == 'good'
    
    return is_positive