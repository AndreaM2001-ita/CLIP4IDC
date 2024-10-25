import os
import json
from openai import OpenAI

from dotenv import load_dotenv

#path to env in google drive
dotenv_path = '/content/drive/MyDrive/env'

# Get openai key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError(
        "API key not found. Please set the OPENAI_API_KEY environment variable.")


client = OpenAI(api_key=OPENAI_API_KEY)

structure = """
   { "name": "v2_100", "attributes":
 [
 { "key": "shape_0", "value": "The oboe in image 1 xxxxxxxx."
 },
 { "key": "color_1", "value": "The oboe in image 1 xxxxxxxx."
 },
 { "key": "texture_2", "value": "The oboe in image 1 xxxxxxxx."
 },
 { "key": "texture_3", "value": "The oboe in image 1 xxxxxxxx."
 }
 ]  """

requirement = f"""please help me transform the file . This file contains comparisons between two images, highlighting their differences. " \
            "Your task is as follows: first, filter out the differences for each img_id in the sentences provided. Categorize these differences into three classes: shape, color, and texture. Remember, these are the only three categories allowed. If there are multiple sentences about shape, label them as shape_0, shape_1, " \
            "shape_2, and so forth, do the same for color and texture. Then, use these labels as keys, and the sentence itself as the value. Note, in the file, the comparisons mention \"before image\" and \"after image.\" You must replace \"before\" with \"image1\" and \"after\" with \"image2.\" " \
          "Ultimately, you should provide me with a JSON array where each item corresponds to an image. The structure of each item in this array should be as follows:{structure}
          Remember, I only need this array returned to me, as I intend to use your message directly to generate the final document, and then please rember
          that I need you to convert entire img_ids ,remember for each img_id ,you need to do all sentences ,secondly please really,just return the json array string,do not add any other of content, otherwise
          how am i supposed to user your message directly, " \
       """


def generateCaptions(filePath):
    assistant = client.beta.assistants.create(
        name="File processing ",
        instructions="You are an expert file processing assistant ",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
    )

    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(
        name="Captioning differences")

    # Ready the files for upload to OpenAI
    file_paths = [filePath]
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    if file_batch.status == "completed":
        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {
                "vector_store_ids": [vector_store.id]}},
        )
    message_file = client.files.create(
        file=open(filePath, "rb"), purpose="assistants"
    )

    # Create a thread and attach the file to the message
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": requirement,
                # Attach the new file to the message.
                "attachments": [
                    {"file_id": message_file.id, "tools": [
                        {"type": "file_search"}]}
                ],
            }
        ]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(
        thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text

    print(message_content.value)

    return message_content.value


def clean_and_convert_json_to_file(json_string):

    # Remove the markdown code block syntax if present
    clean_string = json_string.strip('```json\n')

    # Convert the cleaned string back to a JSON object
    json_object = json.loads(clean_string)

    return json_object


def save_json_to_file(json_object, file_path):
    """
    Saves a JSON object to a specified file path.

    Args:
    json_object (dict or list): JSON object to be saved.
    file_path (str): Path where the JSON file will be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(json_object, file, ensure_ascii=False, indent=4)


def main(input_file_path, output_file_path):
    # Step 1: Generate captions using the provided file
    json_string = generateCaptions(input_file_path)

    # Step 2: Clean the JSON string and convert it to a JSON object
    json_object = clean_and_convert_json_to_file(json_string)

    # Step 3: Save the JSON object to a file
    save_json_to_file(json_object, output_file_path)

    print("The JSON data has been successfully saved to:", output_file_path)


# To execute the main method directly with file paths
if __name__ == "__main__":
    # Define the paths for input and output
    input_path = 'hyp_ep_1.json'  # Update this to your input file path
    output_path = 'after_processed.json'  # Update this to your output file path

    main(input_path, output_path)
