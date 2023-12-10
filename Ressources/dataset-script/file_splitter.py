import json
import os
from pathlib import Path

def save_articles(articles, output_folder, dataset_type, file_count):
    file_name = f'{dataset_type}_part-{file_count}.json'
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w') as outfile:
        json.dump(articles, outfile)
    print(f'File {output_path} created with {len(articles)} articles')

def split_file(input_file, output_base_folder, dataset_type, articles_per_file, max_files=10):
    output_folder = Path(output_base_folder) / dataset_type
    output_folder.mkdir(parents=True, exist_ok=True)

    articles = []
    file_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            if file_count >= max_files:
                break

            try:
                article = json.loads(line)
                articles.append(article)

                if len(articles) == articles_per_file:
                    save_articles(articles, output_folder, dataset_type, file_count)
                    articles = []
                    file_count += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {input_file}, line: {line}")
                print(f"Error message: {e}")

    if articles and file_count < max_files:
        save_articles(articles, output_folder, dataset_type, file_count)

# Utilisation de la fonction
base_path = Path("/Users/samueldorismond/Documents/Cours_BUT3/ALOQAS/Ressources/pubmed-dataset")
output_base = "../chunking-dataset"

split_file(base_path / "test.txt", output_base, "train", 1000, 10)
split_file(base_path / "train.txt", output_base, "test", 1000, 20)
split_file(base_path / "val.txt", output_base, "val", 1000, 10)
