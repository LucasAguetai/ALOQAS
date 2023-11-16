import json
import os

def split_file(input_file, output_base_folder, dataset_type, articles_per_file):
    # Définir le chemin du dossier de sortie spécifique au type de dataset
    output_folder = os.path.join(output_base_folder, dataset_type)

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    articles = []
    file_count = 0

    with open(input_file, 'r') as file:
        for line in file:
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

    if articles:
        file_name = f'{output_folder}/{dataset_type}_part-{file_count}.json'
        with open(file_name, 'w') as outfile:
            json.dump(articles, outfile)
        print(f'File {file_name} created with {len(articles)} articles')

# Exemple d'utilisation
ressouces_folder = '../../Ressources'
base_folder = ressouces_folder + '/chunking-dataset'
split_file(ressouces_folder + '/pubmed-dataset/train.txt', base_folder, 'train', 1000)
split_file(ressouces_folder + '/pubmed-dataset/val.txt', base_folder, 'val', 1000)
split_file(ressouces_folder + '/pubmed-dataset/test.txt', base_folder, 'test', 1000)
