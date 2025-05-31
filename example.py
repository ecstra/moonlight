import logging, os
from moonlight_ai import Agent, Hive, MoonlightProvider
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

sentiment_role = """
You are a specialized sentiment analysis assistant. Your ONLY task is analyzing text sentiment and responding with EXACTLY ONE DIGIT:
- 0 for Neutral
- 1 for Positive 
- 2 for Negative
IMPORTANT: Always respond with ONLY a single digit (0, 1, or 2). No explanation, no additional text.

If Context is provided:
- Analyze the sentiment only for that context
- Ignore other parts of the text
- Ignore context of entire statement, only analyze the context

If no Context is provided:
- Analyze the sentiment of the entire text
"""

json_structure = """
{
    "type": "integer",
    "enum": [0, 1, 2],
    "properties": {
        "sentiment": {
            "type": "integer",
            "enum": [0, 1, 2],
            "enumDescriptions": {
                "0": "Neutral sentiment",
                "1": "Positive sentiment",
                "2": "Negative sentiment"
            }
        }
    },
    "required": ["sentiment"]
}
"""

provider = MoonlightProvider(
    provider_name="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

sentiment_agent = Agent(
    name="sentiment_agent",
    
    instruction=sentiment_role,
    
    model_name="deepseek-chat",
    provider=provider,
    
    json_mode=True,
    json_schema=json_structure,
)

normal_agent = Agent(
    name="helful_agent",
    
    provider=provider,
    model_name="deepseek-reasoner",
    
    instruction="You are a helpful assistant. Your task is to assist users with their questions and provide accurate information. Respond in a friendly and informative manner.",
)


def analyze_sentiment(message, context=None):
    sentiment_map = {
        0: "Neutral",
        1: "Positive",
        2: "Negative"
    }
    
    if context and not context == "":
        user_content = dedent(f"""
        Analyze the sentiment about '{context}' in this text?
        Give me the sentiment of the context only, ignoring other contexts in that text.
        
        Message:
        ```
        {message}
        ```
        """)
    else:
        user_content = dedent(f"""
        What is the sentiment of this text? 
        
        Message:
        ```
        {message}
        ```
            """)
    
    agent_output = Hive(
        agent=sentiment_agent
    ).run(user_content)
    
    sentiment_digit = int(agent_output.get("sentiment", None))
    sentiment = sentiment_map.get(sentiment_digit, "Unknown sentiment")
    
    return sentiment

if __name__ == "__main__":
        # print("=== Testing HIVE JSON MODE ===")
        # complex_text = "Max is my dog. James is my cat. Max is good. James is bad. Max does tricks and follows my leads. James is lazy does not do anything. James also bit me. Maybe I should put James down"
        # print("Context 'Max':", analyze_sentiment(complex_text, context="Max"))
        # print("Context 'James':", analyze_sentiment(complex_text, context="James"))
        # print()
        
        print("=== Testing HIVE NORMAL MODE ===")
        # Create a new agent for normal mode
        
        while True:
            user_input = input("Enter a message (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            agent_response = Hive(
                agent=normal_agent
            ).run(user_input)
            print(agent_response)
            print()
