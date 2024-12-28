import json

def get_chatbot_response(client, model_name, messages, temperature=0):
    """Get response from the chatbot with proper error handling"""
    try:
        input_messages = []
        for message in messages:
            input_messages.append({
                "role": message["role"], 
                "content": message["content"]
            })

        response = client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            temperature=temperature,
            top_p=0.8,
            max_tokens=2000,
        ).choices[0].message.content
        
        return response.strip()  # Clean any whitespace
    except Exception as e:
        print(f"Error in get_chatbot_response: {str(e)}")
        raise

def double_check_json_output(client,model_name,json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(client,model_name,messages)

    return response

def get_embedding(embedding_client, model_name, text_input):
    """Get embeddings with error handling"""
    try:
        output = embedding_client.embeddings.create(
            input=text_input,
            model=model_name
        )
        
        embeddings = []
        for embedding_object in output.data:
            embeddings.append(embedding_object.embedding)

        return embeddings
        
    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")
        raise