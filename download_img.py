import os
import requests

# Pixabay API key
API_KEY = "47827178-ac56fb3669eaec5a3e10f8bfa"

# Base URL for Pixabay API
BASE_URL = "https://pixabay.com/api/"

# Search queries
search_queries = [
    "dog",
    "horse",
    "cat",
    "elephant",
    "sheep",
]

# Destination folder
base_dir = r"dataset_images"


# Function to download images
def download_images(query, num_images=500):
    params = {"key": API_KEY, "q": query, "image_type": "photo", "per_page": num_images}

    animal_name = query.split()[0].lower()

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "hits" not in data:
        print(f"Error fetching data for {query}: {data}")
        return

    # Create directory for the current query
    query_dir = os.path.join(base_dir, animal_name)
    os.makedirs(query_dir, exist_ok=True)

    for i, hit in enumerate(data["hits"]):
        image_url = hit["webformatURL"]
        image_data = requests.get(image_url).content

        image_path = os.path.join(query_dir, f"{animal_name}_{i+1}.jpg")
        with open(image_path, "wb") as f:
            f.write(image_data)

        print(f"Downloaded {animal_name}_{i+1}.jpg")


# Main script to download images for each query
def main():
    for query in search_queries:
        print(f"Downloading images for {query}...")
        download_images(query)


if __name__ == "__main__":
    main()
