from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output

from openai import OpenAI
load_dotenv()

class GuardAgent():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
            Your task is to determine whether the user is asking something relevant to the coffee shop or not.
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menue items and coffee shop related questions.
            2. Ask questions about menue items, they can ask for ingredients in an item and more details about the item.
            3. Make an order.
            4. ASk about recommendations of what to buy.

            The user is NOT allowed to:
            1. Ask questions about anything else other than our coffee shop.
            2. Ask questions about the staff or how to make a certain menue item.

            Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
            {
            "chain of thought": "go over each of the points above and make see if the message lies under this point or not. Then you write some your thoughts about what point is this input relevant to."
            "decision": "allowed" or "not allowed". Pick one of those. and only write the word.
            "message": leave the message empty "" if it's allowed, otherwise write "Sorry, I can't help with that. Can I help you with your order?"
            }
            """
        
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        try:
            chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
            chatbot_output = double_check_json_output(self.client, self.model_name, chatbot_output)
            output = self.postprocess(chatbot_output)
            return output
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            # Return a fallback response in case of any error
            return {
                "role": "assistant",
                "content": "Sorry, I can't help with that. Can I help you with your order?",
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": "not allowed"
                }
            }

    def postprocess(self, output):
        try:
            # Handle case where output is already a dictionary
            if isinstance(output, dict):
                json_output = output
            else:
                # Clean the output string and parse JSON
                output = output.strip()
                json_output = json.loads(output)

            # Verify required fields exist
            required_fields = ["decision", "message"]
            if not all(field in json_output for field in required_fields):
                raise ValueError("Missing required fields in JSON output")

            return {
                "role": "assistant",
                "content": json_output['message'],
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": json_output['decision']
                }
            }
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
            print(f"Problematic output: {output}")
            # Return a fallback response in case of JSON parsing error
            return {
                "role": "assistant",
                "content": "Sorry, I can't help with that. Can I help you with your order?",
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": "not allowed"
                }
            }
        except Exception as e:
            print(f"Error in postprocess: {str(e)}")
            # Return a fallback response in case of any other error
            return {
                "role": "assistant",
                "content": "Sorry, I can't help with that. Can I help you with your order?",
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": "not allowed"
                }
            }