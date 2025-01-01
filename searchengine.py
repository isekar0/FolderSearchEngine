import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import threading

# Initialize the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to image folder
path = r"C:\Users\marti\OneDrive\Pictures\Saved Pictures\cats and cute stuff"



def SearchFolder2(path, search_query, image_size=(224, 224), batch_size=2, top_k=1, num_threads=4):
    def load_and_preprocess_images(image_paths, image_size, results):
        """
        Load and preprocess images for a batch in a separate thread.
        Args:
            image_paths: List of image paths to load.
            image_size: Tuple specifying the size to resize images.
            results: Shared list to store preprocessed images and their paths.
        """
        thread_results = []
        for image_path in image_paths:
            try:
                with Image.open(image_path).convert("RGB") as image:
                    image = image.resize(image_size, Image.BICUBIC)
                    thread_results.append((image, image_path))
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        results.extend(thread_results)
    
    best_prob = 0.0
    best_match_image = []

    valid_extensions = (".jpg", ".png", ".jpeg")
    image_paths = [
        os.path.join(path, image_name) for image_name in os.listdir(path) if image_name.lower().endswith(valid_extensions)
    ]
    num_files = len(image_paths)

    # Compute text embeddings once
    text_inputs = processor(text=search_query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    for i in range(0, num_files, batch_size):
        if i % 10 == 0:
            print(f"Checking file no. {i + 1} out of {num_files}")

        # Batch image paths
        batch_paths = image_paths[i:i + batch_size]

        # Multithreading for image loading
        results = []
        threads = []
        batch_split = max(len(batch_paths) // num_threads, 1)
        for j in range(0, len(batch_paths), batch_split):
            thread = threading.Thread(target=load_and_preprocess_images,
                                      args=(batch_paths[j:j + batch_split], image_size, results))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Extract images and paths from results
        images, processed_paths = zip(*results) if results else ([], [])

        if not images:
            continue

        # Process image features
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(text_features, image_features, dim=-1)

        # Update best matches
        for similarity, image_path in zip(similarities.tolist(), processed_paths):
            if similarity > best_prob:
                best_prob = similarity
                best_match_image.append(image_path)
                if len(best_match_image) > top_k:
                    del best_match_image[0]

    return best_match_image


# # Example usage
# search_query = ["picture of rat"]
# best_image = SearchFolder2(path, search_query, top_k=3, num_threads=12)
# print(best_image)
