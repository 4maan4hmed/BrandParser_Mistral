import json
import data_operation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

items = data_operation.get_items()

# Function to compare OCR text and return the most similar item's number
def compare(text):
    # Extract ocr_data and item_number from each item in the JSON
    ocr_data = {item['item_number']: item['ocr_data'] for item in items}

    # OCR result from the camera (with errors)
    ocr_camera_result = text

    # Combine all OCR data texts
    texts = list(ocr_data.values())
    texts.append(ocr_camera_result)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()

    # Cosine similarity between the camera result and all saved OCR data
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()

    # Find the item with the highest similarity score
    most_similar_index = cosine_similarities.argmax()
    print(most_similar_index)
    most_similar_item_number = list(ocr_data.keys())[most_similar_index]
    similarity_score = cosine_similarities.max()
    
    # Return the item number with the highest similarity
    return most_similar_item_number
# Example usage
ocr_camera_result = "MIST Odonil Lavender Lavende Odoni Odon ROON Lavend 85.5g/150m Laver your home ROOM SPRA ovenderMIST one leel special."
item_number = compare(ocr_camera_result)

# Output result
print(f"The most similar item number is: {data_operation.get_item_details(item_number)["item_name"]}")
print(item_number)
