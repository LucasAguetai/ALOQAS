import json
import os

def split_file(input_file, output_base_folder, dataset_type, articles_per_file, max_files=10):
    # Définir le chemin du dossier de sortie spécifique au type de dataset
    output_folder = os.path.join(output_base_folder, dataset_type)

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    articles = []
    file_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            if file_count >= max_files:
                break  # Arrêter si le nombre maximum de fichiers est atteint

            try:
                article = json.loads(line)
                articles.append(article)

                if len(articles) == articles_per_file:
                    file_name = f'{output_folder}/{dataset_type}_part-{file_count}.json'
                    with open(file_name, 'w') as outfile:
                        json.dump(articles, outfile)
                    print(f'File {file_name} created with {len(articles)} articles')
                    articles = []
                    file_count += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}")
                print(f"Error message: {e}")

    if articles and file_count < max_files:
        file_name = f'{output_folder}/{dataset_type}_part-{file_count}.json'
        with open(file_name, 'w') as outfile:
            json.dump(articles, outfile)
        print(f'File {file_name} created with {len(articles)} articles')