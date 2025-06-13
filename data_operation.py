import json
# Load items from the JSON file
with open('Database/data.json', 'r') as file:
    items = json.load(file)
"""--------------------------------------------------------------------------------------------"""
def get_items():
    return items
def get_item_details(item_number):
    for item in items:
        if item['item_number'] == item_number:
            # Return all the details (item dictionary) for the matched item number
            return item
    return None  # Return None if the item number is not found

"""--------------------------------------------------------------------------------------"""

# Function to add an item with default values
def add_item(item_number, item_name="Unknown", brand="Unknown", barcode=None, weight="0g", price="0/-", ocr_data=None):
    new_item = {
        "item_number": item_number,
        "item_name": item_name,
        "brand": brand,
        "barcode": barcode,
        "ocr_data": ocr_data,  # Set to null
        "weight": weight,
        "price": price
    }
    # Append the new item to the items list
    items.append(new_item)

    # Save the updated list back to the JSON file
    with open('Database/data.json', 'w') as file:
        json.dump(items, file, indent=4)

    print(f"Item {item_number} added successfully.")

"""-------------------------------------------------------------------------------------------------"""
