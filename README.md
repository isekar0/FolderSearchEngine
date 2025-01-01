# FolderSearchEngine

## Overview
A Flask-based web app that performs semantic image searches within a folder using Hugging Face CLIP. Users input an image folder and search query via a form, and the app ranks images based on similarity to the query.

## Features
- Semantic image search using CLIP.
- Batch processing for efficiency.
- Dynamic input via a web form.
- Minimalist HTML/CSS frontend.

## Tech Stack
- Backend: Flask
- ML: Hugging Face transformers (CLIP), PyTorch
- Frontend: HTML, CSS

## Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install flask transformers torch pillow
```
## Setup & Usage
1) Clone the repository
```bash
git clone https://github.com/your-username/huggingface-image-search.git
cd huggingface-image-search
```
2) Run the app
```bash
flask app run
```
3) Access the app at http://127.0.0.1:5000

## Example Workflow
1) Place images in a folder.
2) Launch the app and provide the folder path and search query.
3) View ranked image results.
## Limitations
- Requires sufficient GPU memory.
- Assumes images are in .jpg, .jpeg, or .png.

## Future Enhancements
- Display image previews.
- Add drag-and-drop folder selection.
- Deploy on cloud platforms (e.g., AWS, Heroku).

## Licence
MIT Licence

## Author
Martin Hanna
Github: isekar0
