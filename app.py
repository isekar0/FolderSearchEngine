from flask import Flask, render_template, request
from searchengine import SearchFolder2
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        folder_path = request.form.get("path")
        search_query = request.form.get("search_query")

        try:
            # Check if the folder exists
            if not os.path.exists(folder_path):
                return f"Error: Folder '{folder_path}' does not exist."

            # Call the SearchFolder2 function
            results = SearchFolder2(folder_path, [search_query])
        except Exception as e:
            results = [f"Error: {str(e)}"]

    return render_template("index.html", results=results)

