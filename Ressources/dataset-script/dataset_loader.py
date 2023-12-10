import json
import os

def load_files_from_directory(directory):
    all_data = []
    all_labels = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                articles = json.load(file)
                for article in articles:
                    for label, data in article.items():
                        all_data.append(data)
                        if label not in all_labels:  # S'assurer que le label est unique
                            all_labels.append(label)

    return all_data, all_labels

def load_dataset(base_directory, dataset_type):
    dataset_directory = os.path.join(base_directory, dataset_type)
    return load_files_from_directory(dataset_directory)