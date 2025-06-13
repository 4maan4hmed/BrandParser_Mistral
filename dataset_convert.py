import json

# ======= USER INPUT: Provide your file path here =======
input_path = "item_dataset.json"  # ⬅️ Replace with your actual path
output_path = "converted_chat_dataset.json"
# ========================================================

# Load the original dataset
with open(input_path, "r") as f:
    data = json.load(f)

# Process each OCR line as a separate conversation
chat_data = []

for entry in data:
    for ocr_text in entry["ocr_text_list"]:
        conversation = []

        user_prompt = (
            "Extract only the company name, item name, category, and storage recommendation "
            "from the OCR text below. Respond concisely in JSON format without any explanation or extra text.\n\n"
            + ocr_text
        )

        assistant_response = {
            "company_name": entry["company_name"],
            "item_name": entry["item_name"],
            "category": entry["category"],
            "storage_recommendation": entry["storage_recommendation"]
        }

        conversation.append({"from": "human", "value": user_prompt})
        conversation.append({"from": "gpt", "value": json.dumps(assistant_response, indent=4)})

        chat_data.append(conversation)

# Save the chat-style dataset
with open(output_path, "w") as f:
    json.dump(chat_data, f, indent=2)

print("AMAAN AHMAD 22BEC1179")
