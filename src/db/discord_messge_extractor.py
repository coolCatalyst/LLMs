from typing import Optional
import requests
import json


def discord_message_extractor(channelid: str, token: str, message_limit: Optional[int]):
    """
    Retrieve messages from a Discord channel.

    Args:
        channelid (str): The ID of the Discord channel.
        token (str): The authorization token.
        limit (int, optional): The maximum number of messages to retrieve. Optional.

    Returns:
        list: A list of message IDs and content.

    """
    num = 0
    limit = 10

    headers = {"authorization": token}  # Enter  you own token

    last_message_id = None
    messages = []  # Initialize an empty list to store message content and IDs

    while True:
        query_parameters = f"limit={limit}"
        if last_message_id is not None:
            query_parameters += f"&before={last_message_id}"

        r = requests.get(
            f"https://discord.com/api/v9/channels/{channelid}/messages?{query_parameters}",
            headers=headers,
        )
        jsonn = json.loads(r.text)
        if len(jsonn) == 0:
            break

        for value in jsonn:
            message_content = value["content"]
            message_id = value["id"]
            messages.append([message_id, message_content])
            last_message_id = message_id
            num += 1
            print(f"Retrieving message {num} - ID: {message_id}")
            if message_limit is not None and num >= message_limit:
                break

    print("Number of messages collected:", num)
    return messages
