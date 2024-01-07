# Etude du modèle GPT-2 - Groupe ALOQAS
---
Dans le cadre de cette SAE, nous allons faire usage du modèle GPT-2, un Large Language Model (LLM) mis au point par OpenAI.

Le modèle génère du texte en se basant sur le contexte et le contenu textuel précédent celui ci. Son principe se rapproche des réseaux de neurones RNN (Recurrent Neural Network) que l'on a pu étudié auparavant au sein de notre formation, ajustant ses paramètres en se basant sur les résultats obtenus aux couches précédentes. Pour cela, le modèle est pré-entrainé sur un large corpus de texte.

A travers ce notebook, nous nous sommes fixés comme objectif de découvrir le modèle GPT-2, d'étudier les différents modules/modèles associés mis à notre disposition par l'API Keras + Tensorflow, de tester ses capacités de génération de texte dans divers cas d'utilisations, d'établir un constat sur les forces, faiblesses et limites de celui-ci.

## Installation de KerasNLP
---
Pour nos tests sur le modèle GPT-2, nous faisons usage de KerasNLP, une librairie utilisée pour le Natural Language Processing.

Elle constitue une extension de Keras qui est une API de Deep Learning associé à la plateforme de machine learning TensorFlow.

[A propos de Keras NLP](https://keras.io/keras_nlp/)

[A propos de Keras](https://keras.io/about/)


```python
!pip install keras-nlp --upgrade
```

    Collecting keras-nlp
      Downloading keras_nlp-0.6.3-py3-none-any.whl (584 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m584.5/584.5 kB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting keras-core (from keras-nlp)
      Downloading keras_core-0.1.7-py3-none-any.whl (950 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m950.8/950.8 kB[0m [31m46.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (1.4.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (1.23.5)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (23.2)
    Requirement already satisfied: regex in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (2023.6.3)
    Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (13.6.0)
    Requirement already satisfied: dm-tree in /usr/local/lib/python3.10/dist-packages (from keras-nlp) (0.1.8)
    Collecting tensorflow-text (from keras-nlp)
      Downloading tensorflow_text-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m99.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting namex (from keras-core->keras-nlp)
      Downloading namex-0.0.7-py3-none-any.whl (5.8 kB)
    Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from keras-core->keras-nlp) (3.9.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras-nlp) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras-nlp) (2.16.1)
    Requirement already satisfied: tensorflow-hub>=0.13.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow-text->keras-nlp) (0.15.0)
    Collecting tensorflow<2.16,>=2.15.0 (from tensorflow-text->keras-nlp)
      Downloading tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (475.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m475.2/475.2 MB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras-nlp) (0.1.2)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.6.3)
    Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (23.5.26)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.5.4)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.2.0)
    Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (16.0.6)
    Requirement already satisfied: ml-dtypes~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.2.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.3.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.20.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (67.7.2)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (4.5.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.34.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.59.2)
    Collecting tensorboard<2.16,>=2.15 (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp)
      Downloading tensorboard-2.15.1-py3-none-any.whl (5.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.5/5.5 MB[0m [31m69.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorflow-estimator<2.16,>=2.15.0 (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp)
      Downloading tensorflow_estimator-2.15.0-py2.py3-none-any.whl (441 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m442.0/442.0 kB[0m [31m48.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting keras<2.16,>=2.15.0 (from tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp)
      Downloading keras-2.15.0-py3-none-any.whl (1.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m99.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.41.3)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2.17.3)
    Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.0.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.5.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.0.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (1.3.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2023.7.22)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (2.1.3)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (0.5.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<2.16,>=2.15.0->tensorflow-text->keras-nlp) (3.2.2)
    Installing collected packages: namex, tensorflow-estimator, keras, keras-core, tensorboard, tensorflow, tensorflow-text, keras-nlp
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.14.0
        Uninstalling tensorflow-estimator-2.14.0:
          Successfully uninstalled tensorflow-estimator-2.14.0
      Attempting uninstall: keras
        Found existing installation: keras 2.14.0
        Uninstalling keras-2.14.0:
          Successfully uninstalled keras-2.14.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.14.1
        Uninstalling tensorboard-2.14.1:
          Successfully uninstalled tensorboard-2.14.1
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.14.0
        Uninstalling tensorflow-2.14.0:
          Successfully uninstalled tensorflow-2.14.0
    Successfully installed keras-2.15.0 keras-core-0.1.7 keras-nlp-0.6.3 namex-0.0.7 tensorboard-2.15.1 tensorflow-2.15.0 tensorflow-estimator-2.15.0 tensorflow-text-2.15.0
    


```python
!pip install git+https://github.com/keras-team/keras-nlp.git -q
```

      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
      Building wheel for keras-nlp (pyproject.toml) ... [?25l[?25hdone
    


```python
import os

# Le backend de Keras pouvant supporter jax, tensorflow ou pytorch,
# nous choisissons Tensorflow pour être en accord avec le sujet.
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_nlp
import tensorflow as tf
import keras_core as keras
import time
```

    Using TensorFlow backend
    

Nous avons expérimenté avec le code proposé sur Keras afin de réaliser des premiers tests pour découvrir le modèle GPT-2.

[GPT2 Text Generation with KerasNLP](https://keras.io/examples/generative/gpt2_text_generation_with_kerasnlp/)



## Modèle GPT-2
---
### Généralités
Le modèle GPT-2 est representé par GPT2Backbone, qui est l'architecture de base modulable pour toute utilisation souhaitée. Pour la génération de texte, nous allons utiliser GPT2CausalLM qui est une extension de GPT2Backbone plus orienté vers le langage causal, c'est à dire se basant sur les mots précédents dans la phrase courante pour la production de résultat à la suite.

Il existe différents presets de modèles de GPT-2 composés d'un nombre différents de couches et donc de paramètres.

Pour débuter notre étude, nous utilisons gpt2_base_en qui est un modèle de 12 couches pré-entrainé sur du texte.
Il existe aussi d'autres models tels que "gpt2_medium_en", "gpt2_large_en", "gpt2_extra_large_en", faisant usage d'un nombre plus important de couches et de paramètres.

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABaAAAAJcCAYAAAAPVwMkAAAAAXNSR0IArs4c6QAAAJZlWElmTU0AKgAAAAgABAEaAAUAAAABAAAAPgEbAAUAAAABAAAARgEoAAMAAAABAAIAAIdpAAQAAAABAAAATgAAAAAAAACQAAAAAQAAAJAAAAABAASShgAHAAAAEgAAAISgAQADAAAAAQABAACgAgAEAAAAAQAABaCgAwAEAAAAAQAAAlwAAAAAQVNDSUkAAABTY3JlZW5zaG90ZRj/SQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAddpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDYuMC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iPgogICAgICAgICA8ZXhpZjpQaXhlbFlEaW1lbnNpb24+NjA0PC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjE0NDA8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpVc2VyQ29tbWVudD5TY3JlZW5zaG90PC9leGlmOlVzZXJDb21tZW50PgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KAB/SAwAAABxpRE9UAAAAAgAAAAAAAAEuAAAAKAAAAS4AAAEuAAD/9fV3QLYAAEAASURBVHgB7J0J3E3F/8e/kbZ/lkoRbRQiRSIqLZYSkX1LtmQnZEsqS6FIyC5CSGTft0j2tajIkoo2KZUW6tfv5z/f0feYc+6Ze+9z731wn/MZL885Z86cOTPvmTNz53PmfOe8k8oRHAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAgkmMB5EKATTBTRgQAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIaAIQoFERQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEUoUABOhUwYpIQQAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEIECjDoAACIAACIAACIAACIAACIAACIAACIAACIAACIAACKQKAQjQqYIVkYIACIAACIAACIAACIAACIAACIAACIAACIAACIAACECARh0AARAAARAAARAAARAAARAAARAAARAAARAAARAAARBIFQIQoFMFKyIFARAAARAAARAAARAAARAAARAAARAAARAAARAAARCAAI06AAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgkCoEIECnClZECgIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAEadQAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQCBVCECAThWsiBQEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAACNOoACIAACIAACIAACIAACIAACIAACIAACIAACIAACIBAqhCAAJ0qWBEpCIAACIAACIAACIAACIAACIAACIAACIAACIAACIAABGjUARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAgVQhAAE6VbAiUhAAARAAARAAARAAARAAARAAARAAARAAARAAARAAAQjQqAMgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAKpQgACdKpgRaQgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIQoFEHQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEUoUABOhUwYpIQQAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEIECjDoAACIAACIAACIAACIAACIAACIAACIAACIAACIAACKQKAQjQqYIVkYIACIAACIAACIAACIAACIAACIAACIAACIAACIAACMQsQPfq1Qv0QAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQIB69OjhSwECtC8WeIIACIAACIAACIAACIAACIAACIAACIAACIAACIAACERLAAJ0tKQQDgRAAARAAARAAARAAARAAARAAARAAARAAARAAARAIEUEIECnCBcCgwAIgAAIgAAI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)

### Modules
GPT-2 possède plusieurs modules sous forme de modèles pré-entrainés, chacuns avec leur propre utilité. Parmi ces modèles, nous pouvons citer le modèle de base GPT2Backbone, le modèle GPT2CausalLM, le tokenizer, les préprocesseurs respectifs.

- `[keras_nlp.models.GPT2Backbone]` - Le modèle de base constituant GPT-2 et son architecture de base. Il est généré avec des paramètres aléatoire et agit comme une base personnalisable selon les besoins.

- `[keras_nlp.models.GPT2CausalLM]` - Le modèle représentant une extension de GPT-2 Backbone, orienté sur le langage causal et permettant la génération de texte. Ce dernier possède une méthode generate() qui va nous permettre de fournir un contenu textuel en entrée et de demander une complétion de ce texte via le modèle GPT-2.

- `[keras_nlp.models.GPT2Tokenizer]` - Le tokenizer transforme le contenu textuel, les chaines de caractères fournies en entiers indices pour chaque séquence de caractères et cela par rapport à leur position dans la phrase.

- `[keras_nlp.models.GPT2Preprocessor]` - Associé à GPT2 Backbone, il permet de rajouter une couche de traitement avant le réseau de neurones principal. Cette couche effectue notamment un tokenizing (expliqué précédemment) et la construction d'un dictionnaire d'identifiants.

- `[keras_nlp.models.GPT2CausalLMPreprocessor]` - Ce préprocesseur est destiné à être utilisé en conjonction avec GPT2 Causal LM, et sert à introduire une couche de traitement du texte avant qu'il soit traité par le réseau de neurones du modèle GPT2.

[A propos des différents modules de GPT-2](https://keras.io/api/keras_nlp/models/gpt2/)


## Initialisation du modèle GPT-2 Causal LM
---
GPT2 Causal LM est un modèle de langage causal comme le nom l'indique (LM).

Par définition, son objectif est de prédire les tokens, c'est à dire les mots suivants, en se basant sur ceux précédant sa position actuelle, donc les mots précédents dans la phrase.

Comme mentionné précedemment, il utilise comme modèle de base GPT2Backbone et agit comme une extension de celle-ci pour la génération de texte par langage causal.

Pour l'instant, nous nous intéressons au modèle gpt2_base_en qui est composé de 12 couches.


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)

gpt2_lm_medium = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

```

    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/vocab.json
    [1m1042301/1042301[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/merges.txt
    [1m456318/456318[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/model.h5
    [1m497986112/497986112[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 0us/step
    

Pour notre expérimentation sur le modèle GPT-2, nous avons réfléchi à plusieurs cas de tests :
- **Tests de génération de texte proposés sur keras.io**, fournis dans la documentation sur la génération de texte avec GPT-2 basé sur du texte en entrée
- **Tests personnels**, que nous proposons
  - **Complétion de texte** : comme pour les tests sur keras.io, afin de voir ce que le modèle génère comme texte avec un sujet en entrée

  - **Traitement d'opérations mathématiques** : même si le modèle est focalisé sur la génération de texte, proposer des calculs mathématiques nous permet de voir si le modèle est capable d'adopter une réflexion logique, et si ce dernier a des connaissances dans basiques dans un domaine particulier, ici donc des mathématiques

  - **Questions générales** : à travers ce test, nous souhaitons voir comment le modèle réagit à des questions simples sur des domaines de la vie courante, sans implication de technicité. Ces tests nous permettent également de voir comment le modèle réagit à une situation de question-réponse, ce dernier étant principalement concentré sur la simple génération / complétion de texte

  - **Questions liés à divers sujets liés aux articles scientifiques** : étant donné que l'on souhaite créer un chatbot orienté sur les articles scientifiques, effectuer des tests de génération en posant des questions sur des sujets techniques dans le domaine médical ou des mathématiques avancés (extraits de la dataset associée) nous permet d'évaluer les connaissances du modèle

  Afin d'approfondir ces tests, nous allons effectuer ces tests de génération sur plusieurs presets de modèles, c'est-à-dire en essayant la génération sur le modèle de base à 12 couches, puis le modèle "medium" à 24 couches, ainsi de suite.
  
  Cela nous permettra de comparer les résultats et de constater la différence de performance, de précision et de clareté entre les différents modèles à nombre de couches élevés et moins élevés.

## Tests de genération de texte sur GPT2 Base
---

### Tests proposés sur keras.io


```python
start = time.time()

output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    My trip to Yosemite was a whirlwind of adventure, with many people I met, and I was surprised to learn that many of the people I met were also in Yosemite. This was my first time in Yosemite and I was thrilled to learn about many of the people who had made the trip.
    
    I was also amazed at how well the park was doing. I was very surprised that there was so much more to see than just the view. There was so much to see that I was amazed that there was so many people who had visited and been inspired by the park. It was a very unique time to be in Yosemite and I am very proud to say that the experience was so much more than a day trip. The park was filled with amazing wildlife, beautiful scenery, amazing food, amazing people, and a great sense of community. I am very grateful to everyone who came to see us, especially the people who had been here for so long.
    
    I also learned how many people
    TOTAL TIME ELAPSED: 0.78s
    


```python
start = time.time()

output = gpt2_lm.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    That Italian restaurant is now open for business, but it's not just about a new menu or a new restaurant.
    
    It's about a new restaurant that's already been opened.
    
    The Italian-American restaurant in the former Italian Quarter is named "La Vida della Siena," which translates literally "Italian Restaurant of the Week."
    
    The restaurant, located at 711 W. Broadway, is a new concept that will feature an open kitchen, an open menu, a new menu, and a new menu.
    
    The menu will include a menu of dishes like chicken, beef, and lamb and a new menu that will include a new menu, a new menu. The new menu will include a new menu.
    
    "We're going to be open every day of the week, and we're going to have a new menu every day of the week," said restaurant manager, Maria Piazza.
    
    Piazza said the new menu will include
    TOTAL TIME ELAPSED: 1.54s
    

### Tests personnels

#### Test complétion de texte


```python
start = time.time()

output = gpt2_lm.generate("Working on this AI development homework,", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    Working on this AI development homework, I'm thinking about how the AI might be able to make it possible to create a game that can be easily played by a group of people. I'd like to start with the simplest example, and then try to get a feel for the potential of the AI to be a game, and what the potential is of a team working in the field.
    
    
    This is not a complete list, as I'm not sure how much more complex it is. The first example, which I've already written up, is about a simple game where players play a group of characters, who are all in a single room, and each other. The AI is a game, and you can play it as a group, or in a multiplayer game. I'm going to assume that you have a few players playing the group and some players playing the group. I'm going to assume that there are two groups, one for each group of characters, and one for each group
    TOTAL TIME ELAPSED: 2.06s
    

#### Test traitement d'opérations mathématiques


```python
start = time.time()

output = gpt2_lm.generate("I wish to calculate the mathematical addition of four plus six, which results to ", max_length=100)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    I wish to calculate the mathematical addition of four plus six, which results to  (1+6)=4+12.
    This is a simple example of a simple calculation:
    1+4 = 2+4.
    This is the same as the previous example.
    I have used the following formula:
    (4+4) = 2+4.
    Now let's see what this means for the number of digits.
    This means that the number of numbers in the
    TOTAL TIME ELAPSED: 71.30s
    


```python
start = time.time()

output = gpt2_lm.generate("1 + 1", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    1 + 1 + 1 + 1 + 1
    
    2 + 1 + 1 + 0
    
    3 + 1 + 1 + 3
    
    4 + 1 + 1 + 1
    
    5 + 1 + 1 + 3
    
    6 + 1 + 1 + 3
    
    7 + 1 + 1 + 3
    
    8 + 2 + 1 + 1
    
    9 + 1 + 1 + 3
    
    10 + 1 + 1 + 2
    
    11 + 1 + 1 + 2
    
    12 + 2 + 1 + 1
    
    13 + 2 + 1 + 1
    
    14 + 3 + 1 + 1
    
    15 + 3 + 1 + 1
    TOTAL TIME ELAPSED: 54.00s
    

#### Tests questions générales


```python
start = time.time()

# 1ère génération
output = gpt2_lm.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  the automobile consists of a motor, a motor is a piece of machinery that is used to carry a load, it is an automobile, it is an automobile. The automobile is a piece of machinery that carries a load. The car is an automobile. The automobile is an automobile. The automobile is an automobile. The vehicle is an automobile. The vehicle is an automobile. The vehicle is an automobile. The vehicle is an automobile. It has to carry loads. It has to carry loads. It is a piece of machinery, a piece of machinery that carries a load, it is an automobile.  It is a piece of machinery that carries a load. It is a piece of machinery that carries a load. It is a piece of machinery that carries an load. It is a piece of machinery that carries an load. It is a piece of machinery that carries a load. It is
    TOTAL TIME ELAPSED: 28.41s
    


```python
start = time.time()

# 2ème génération
output = gpt2_lm.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=100)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  I don't know. I am a car enthusiast and I have been using a lot of different brands of cars over the last few years. The most common ones being Ford Fiesta, Chevrolet Malibu, and Mercedes-Benz C63. I think the most common car I use is the Toyota Prius with the exception of a couple of my friends who own a Toyota Prius. I have
    TOTAL TIME ELAPSED: 13.46s
    


```python
start = time.time()

# 1ère génération
output = gpt2_lm.generate("What is the capital city of the country France ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    What is the capital city of the country France ?
    
    It is a small country of 1m people.
    
    How do you make it?
    
    The capital city of France has a population of about 1.6 million.
    
    How do you get there, and why?
    
    The French are not very good people. They are not very good people. They are not very good citizens.
    
    What are the reasons for the decline in their quality of life ?
    
    They have become poorer and more miserable. They don't have any kind of job. The French are not very good people. They are not very good people. They are not very good people.
    
    How did you come to be in this country ?
    
    I came here in the early 1990s. I went to school. I went to university. I went to school for two years and then got my PhD. I worked at the local railway station and then went to the university. Then
    TOTAL TIME ELAPSED: 30.38s
    


```python
start = time.time()

# 2ème génération
output = gpt2_lm.generate("What is the capital city of the country France ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    What is the capital city of the country France ?
    
    The capital city is Paris.
    
    How many people live in Paris?
    
    There are 1.5 million people in Paris, and there are 2.5 million people in the rest of Europe.
    
    What is the capital city of Germany?
    
    There is no capital city, but the capital of Berlin.
    
    How many people live in France?
    
    There are 3,200,000 people, and there are 2 million people in France.
    
    What is the capital city of Italy?
    
    There is a capital city, Florence, which lies in Italy.
    
    What are the capital cities of Germany?
    
    There are 4,000,000 people in Germany, and there are 2.1 million people in France.
    
    What is the capital city of France?
    
    There is a city, Nice.
    
    How many people live in France?
    
    There are 2,
    TOTAL TIME ELAPSED: 33.51s
    

#### Tests lié à divers sujets de la dataset scientific papers
On observe que lorsqu'on interroge le modèle GPT-2 sur divers sujets traités dans des articles scientifiques, le modèle arrive à reprendre les termes du sujet fournies et d'utiliser le vocabulaire associé à celui-ci, et parvient même parfois à comprendre le sujet en question (exemple de l'hypertrophie cardiaque, où le modèle a mentionné le manque ou l'abondance d'oxygène qui influence les crises cardiaques).

Cependant, on constate que les phrases n'ont peu voir aucune signification et ont tendance à être juste une répétition des termes techniques du vocabulaire. Cela est le cas notamment dans les deux dernières réponses (multiplex et déficit héréditaire en facteur de coagulation), où en l'absence à priori de connaissances du modèle sur le sujet, il se contente de répéter les termes techniques tels "multiplex", "multi-x" et "coagulation factor", "coagulation protein". Le sujet au final n'est pas expliquée en détail, voir n'est pas abordé.


```python
start = time.time()

# 1ère génération
# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm.generate("I've been given the text 'cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues1'. \
  On this topic, I know that", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    I've been given the text 'cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues1'.   On this topic, I know that the 'cardiac hypertrophy' hypothesis is a bit off the mark.
    
    In this article, I'll discuss how my own research has shown how my heart is not as healthy as people think it is.
    
    What I find most interesting about this hypothesis is that it seems to be based on a very small group of people.  It seems to me that the majority of people who believe that heart failure is caused by a lack of oxygen are either ignorant of the scientific literature or just ignorant of the concept of 'hypertrophy'.  I've seen people who believe that a high level of oxygen in the blood is responsible for a large part of the heart's failure, but who are unaware about the role that the oxygen plays.  I also have seen
    TOTAL TIME ELAPSED: 26.56s
    


```python
start = time.time()

# 2ème génération
# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm.generate("I've been given the text 'cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues1'. \
  On this topic, I know that", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    I've been given the text 'cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues1'.   On this topic, I know that there is an underlying biological basis for the hypertrophy, but the concept does not hold up to scrutiny.
    
    It's not as if this is a new phenomenon. The concept is common in neurology and has been in the literature for many years.
    
    There are several different definitions of what is hypertrophy. The first is a physiological condition where the body produces a high-energy, hypertrophy signal, which then accelerates. This is termed hypertrophy hypertrophy.
    
    The second definition of hypertrophy is a pathological condition where the body produces a low energy, hypertrophy signal, which continues to accelerate until death.
    
    This term is used to describe a condition when there is a sudden and significant reduction in the energy of the body
    TOTAL TIME ELAPSED: 28.89s
    


```python
start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm.generate("How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    
    The answer is that the structure of a complex multi-x is the key, and it's not just the structure of a complex multiplex. In the real world, there are two main types of multi-x: the complex multi-x, the complex multix, and the complex multiplex that is used to make the walk applications.
    
    The complex multi-x is an abstraction of the multi-x. It has the properties of a single complex. It has the properties that a multi-x can be used to do, and it has properties of a complex multiplex.
    
    The complex multi-x can be used in a variety of ways, and it's a very important abstraction of the multi-x. It has the properties of a multiplex that is used to make the walk applications.
    
    In the real world, it has the properties that
    TOTAL TIME ELAPSED: 30.14s
    


```python
start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository pubmed
output = gpt2_lm.generate("Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ? I am not a doctor.
    
    The Inherited defect of coagulation factors is a common condition of all coagulation factors, but is also a condition of the immune system, the immune system is a part of the body that is affected by all these factors.
    
    I am very happy that I am able to explain to you how to explain the Inherited defect of fibrinolysis, and the Inherited defect of PE and the Inherited defect of coagulation factors.
    
    What is coagulation factor X?
    
    Coagulation factor X is the genetic condition of fibrosarcoma.
    
    Coagulation factor X is a condition of fibrosarcoma. Coagulation factor X is caused by a mutation in the gene for the protein fibrin, the enzyme that is responsible
    TOTAL TIME ELAPSED: 85.09s
    

## Observations suite à nos premiers tests sur le modèle de base
---
On peut constater que GPT-2 arrive bien à générer du texte en rapport avec ce qu'on lui a fourni. Cependant, il reste limité à la génération de texte, et on ne peut lui donner de véritable questions (comme des opérations mathématiques).

Le modèle à ce stade est très générique et ne se focalise que sur la génération de texte visant à compléter un texte fourni en entrée. Le résultat est aléatoire et parfois cohérent, d'autres fois illogique et contradictoire.

Pour répondre à ces problèmes et améliorer le modèle ainsi que ses réponses, nous pouvons adopter une approche par fine-tuning. L'objectif va être de créer un nouveau modèle mobilisant celui de GPT-2, et de l'entraîner sur un sujet particulier avec un dataset spécialisé dans un domaine. Il sera donc possible de passer d'un modèle générique à un modèle plus spécifique.

En lui fournissant également une dataset spécialisé dans un domaine, un sujet particulier, notre modèle sera capable de répondre plus précisément aux questions, dans notre cas concernant des articles scientifiques (scientific papers).

### Forces
- Capable de générer du texte en essayant de garder une certaine cohérence
- Produit des phrases structurés et grammaticalement correctes
- Quand le contexte est bien fourni, le modèle arrive à saisir quel sujet et domaine est traité

### Faiblesses et limites
- Se contredit dans les propos générés
  - incohérence, contradiction
- Complétion aléatoire, phrases résultantes sans signification
  - répétition de termes
- Résultats parfois subjectives
  - le modèle est susceptible de fournir des réponses personnifiés avec une situation imaginaire, fictive
  - exemple de la question générale sur la capitale de la France : s'imagine comme quelqu'un qui a étudié en France en 1990, et qui a une opinion négative et subjective sur le sujet
  - exemple de la question générale sur ce qui constitue une voiture, le modèle fournit une réponse en se décrivant comme un fan de voiture qui a conduit plusieurs voitures spécifiques, mais ne connait pas les éléments d'une voiture
- Difficulté de répondre à des questions concrètes et à effectuer des opérations mathématiques
  - le modèle est focalisé sur de la complétion de texte, et moins sur une forme de question-réponse
- Connaissances limités
  - les réponses ne sont que de surface et ne traitent pas en profondeur le sujet
  - exemple : domaine des mathématiques ou médical

## Tests de genération de texte sur GPT2 Medium
---
### Initialisation du modèle gpt2_medium_en


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_medium_en",
    sequence_length=128,
)

gpt2_lm_medium = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_medium_en", preprocessor=preprocessor
)
```

    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_medium_en/v1/vocab.json
    [1m1042301/1042301[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_medium_en/v1/merges.txt
    [1m456318/456318[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_medium_en/v1/model.h5
    [1m1419729400/1419729400[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m18s[0m 0us/step
    

### Tests proposés sur keras.io


```python
start = time.time()

output = gpt2_lm_medium.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My trip to Yosemite was a success.
    
    I had planned on spending a day in the park and a few nights exploring the surrounding areas. The plan worked.
    
    But, the day I went, I was not prepared for the incredible amount of fun and excitement that ensued.
    
    The park was filled with all manner of wild animals and wild plants.
    
    I was so overwhelmed that I didn't even know what to do.
    
    I was in a constant state of confusion as to what to do next.
    
    I was in awe at the incredible variety of wildlife I could see.
    
    The park was a wonderful place to spend an amazing day.
    
    But, as I continued to explore, my excitement was quickly replaced by confusion.
    
    The park was so vast that I could not find the exact location of each of the many trails.
    
    I tried to find the best trail, the most scenic one, or the most scenic spot to explore.
    TOTAL TIME ELAPSED: 214.11s
    


```python
start = time.time()

output = gpt2_lm_medium.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    That Italian restaurant is now closed, but the owners say they're still working hard to reopen.
    
    Italian restaurant owner Mario Bocca has been working on the restaurant's website for the past month, and it was recently announced that his restaurant would be open in early November.
    
    But the restaurant is closed now, and Bocca says he's working with a team of about 20 to help him reopen.
    
    Bocca says the restaurant will be open on the weekends from 11 a.m. to 4 p.m., and will be open on Sundays from 11 a.m. to 4 p.m.
    
    He hopes to reopen in early November, but he says it will take at least two months before the restaurant can be ready for its new owners.
    TOTAL TIME ELAPSED: 4.05s
    

### Tests personnels

#### Test complétion de texte


```python
start = time.time()

output = gpt2_lm_medium.generate("Working on this AI development homework,", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    Working on this AI development homework, I've decided to make it into an actual game that I could play, but I'd like to make the AI play as well. The main goal is to find the optimal way to use a certain resource.
    
    The AI is currently in a state of panic. It has no idea what to do, and it's trying to figure out what to do. It has to figure out how to get the resources it's missing and then decide if it wants those resources or not. It has to figure out how to make its next move and if it can get the resources it needs. The AI is trying to find the best way to use resources, and it can't seem to figure it out.
    
    So what should be the AI's goal? I'm thinking that it should try to figure out how to find the resources that it doesn't have, and then figure out a way to get them back. If it can't find them, then
    TOTAL TIME ELAPSED: 122.07s
    

#### Test traitement d'opérations mathématiques


```python
start = time.time()

output = gpt2_lm_medium.generate("I wish to calculate the mathematical addition of four plus six, which results to", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    I wish to calculate the mathematical addition of four plus six, which results to a sum of two.
    
    If we take four and add six, we get two:
    
    I wish to calculate the mathematical add of four plus six, which results to a sum of two, and I wish to calculate the mathematical subtract of four from six.
    
    I wish to calculate the mathematical subtraction of four from six, which results to the sum of four minus six.
    
    The sum of the three numbers is:
    
    I want to calculate the mathematical addition of two plus two, which result to the sum of two plus four.
    
    I wish to calculate the mathematical subtraction of two from two and four, which results to the sum of two minus two and four.
    
    I wish to calculate the mathematical addition of six plus six, which results to the sum of four plus four.
    
    I wish to calculate the mathematical subtraction of six from six, which result to
    TOTAL TIME ELAPSED: 114.55s
    

#### Test question générale


```python
start = time.time()

output = gpt2_lm_medium.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  "  a motor vehicle  is a vehicle which uses electricity  to move about."  In short, it consists of a motor, which is an engine that uses electricity to propel a motor.  An electric motor is a motor which uses electricity to propel itself.  In a nutshell :  electric motors are vehicles which can be charged via the use of an electric current.  In other words :  electric motors use electricity to move.  An automobile uses electricity to move.  An automobile consists of two parts - a motor and an engine.
    In the previous question, the term  "electric motor" was used to refer to an engine.  The electric motor uses electricity to move itself.  An automobile, on the other hand, consists of two parts - a motor and an engine.  An
    TOTAL TIME ELAPSED: 207.63s
    

#### Tests lié à divers sujets de la dataset scientific papers


```python
start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm_medium.generate("How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

print("=======================================")

start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository pubmed
output = gpt2_lm_medium.generate("Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    
    In this article, I will explain how the structure of multiplexed transport networks affects how a walk application is able to find and navigate through a network of multiplexes. I will then discuss some examples of how this structure can be used to improve the walk applications performance.
    
    I have chosen to focus on a network of multiplexes for two reasons. First, I believe that multiplexes are a key aspect to the performance of walk applications. Second, multiplexes are a critical component to the performance of multiplexed transport networks. I will discuss how multiplexes can be used to optimize the performance of the network and improve the walk performance.
    
    I will also discuss some examples of how multiplexes can be used to optimize the performance of a multi-pass walk application.
    
    The structure of multiplexed transportation networks
    
    The structure of multiplexes has two major components
    TOTAL TIME ELAPSED: 108.29s
    =======================================
    
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    
    I think that the PE Inherited defects of coagulation factors and PE inherited defects of fibrinolysis are related to one another, because PE inherited defects of fibrinoside are also caused by PE inherited defects of fibrinolysis and coagulation factors are not related to one another.
    
    In the case of PE, it is not the same thing as coagulation factors, it is just different in the two cases. The reason for the difference is that PE inherited defects of fibrinolysis and fibrinoside have different properties.
    
    In the case of PE, coagulation factors are not the same as PE inherited defects.
    
    What is the mechanism by which PE Inherited defects of fibrinolysis and fibrinolysis can cause coagulation defects?
    TOTAL TIME ELAPSED: 103.42s
    

## Tests de genération de texte sur GPT2 Large
---
### Initialisation du modèle gpt2_large_en


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_large_en",
    sequence_length=128,
)

gpt2_lm_large = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_large_en", preprocessor=preprocessor
)
```

    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_large_en/v1/vocab.json
    [1m1042301/1042301[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_large_en/v1/merges.txt
    [1m456318/456318[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step       
    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_large_en/v1/model.h5
    [1m3096768960/3096768960[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 0us/step
    

### Tests proposés sur keras.io


```python
start = time.time()

output = gpt2_lm_large.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My trip to Yosemite was a great way to get to know the park and the people who live there, but it didn't do much to make me appreciate the park.
    
    I'm not sure if it's because it's such a small park or because I'm used to the park being big. But the view from the summit of Half Dome, for example, is pretty spectacular. I was also surprised that I was able to see some of the best views in the valley.
    
    Advertisement
    
    Advertisement
    
    Advertisement
    
    Advertisement
    
    Advertisement
    
    Advertisement
    
    I was able to see the best views of the valley in the middle of a storm and it made the trip worthwhile.
    
    Advertisement
    
    I was able to see Yosemite's famous Half Dome, a huge rock that's the highest point on the mountain. I also got a glimpse of the valley below the top. I had a great view of Yosemite's famous Half Dome. I was also able to see
    TOTAL TIME ELAPSED: 248.45s
    


```python
start = time.time()

output = gpt2_lm_large.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    That Italian restaurant is a bit of a mystery. It is located right in the centre of the city, but there is no sign. It is also the only place that is open on a Saturday morning.
    
    But what is the secret of this hidden gem?
    
    The restaurant was built in the 1950s by a family from Italy who lived in the town of Tuscany and had a restaurant in the old town. The restaurant is located at the corner of the road that leads down from the Tuscany town center to the town's old town, and the family had been living in this area for over 100 years.
    
    The restaurant is a little hidden gem, but it's a very nice one. The restaurant is open on a Saturday morning and it has the most beautiful decor. It is very quiet and there is a lot of seating inside, so there is plenty to do if you want to relax and have some food.
    
    The restaurant is also located in the
    TOTAL TIME ELAPSED: 4.85s
    

### Tests personnels

#### Test complétion de texte


```python
start = time.time()

output = gpt2_lm_large.generate("Working on this AI development homework,", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    Working on this AI development homework, I was inspired to try something new.
    
    I was thinking how I can make a simple AI that learns how to play the game, and then I thought:
    
    How can we make it learn from experience?
    
    I started to think about how to make it learn from experience, and I came across this post by @paulkremer on how to do it with reinforcement learning.
    
    I thought it would be cool to try to use reinforcement learning to make a simple AI learn from experience.
    
    This AI is a simple one, but it can still teach itself to learn how to play.
    
    The AI will play a game and try to learn how to play.
    
    Here's an example of the AI learning how to play.
    
    Here's what it's trying to learn:
    
    I used this code as my starting point. It is a simple example.
    
    #!/usr/bin/env
    TOTAL TIME ELAPSED: 5.81s
    

#### Test traitement d'opérations mathématiques


```python
start = time.time()

output = gpt2_lm_large.generate("1 + 1", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    1 + 1 1 = 4
    
    The number 1 + 1 1 = 5
    
    The number 1 + 1 1 = 6
    
    The number 1 + 1 1 = 7
    
    The number 1 + 1 1 = 8
    
    The number 1 + 1 1 = 9
    
    The number 1 + 1 1 = 0
    
    The number 1 + 1 1 = 0.1
    
    The number 1 + 1 1 = 0.01
    
    The number 1 + 1 1 = 0.02
    
    The number 1 + 1 1 = 0.03
    
    The number 1 + 1 1 = 0.04
    
    The number 1 + 1 1 = 0.05
    
    The number 1 + 1 1 = 0.06
    
    The number 1 + 1 1 = 0.07
    
    The number 1 + 1 1 = 0.08
    
    The number 1 + 1 1 = 0.09
    
    The number 1 + 1 1 = 0.10
    
    TOTAL TIME ELAPSED: 4.90s
    

#### Test question générale


```python
start = time.time()

#1ère génération
output = gpt2_lm_large.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  a vehicle is made up of a number of parts and the parts are interconnected by a number of connections.  The connection is a mechanical device that is used to control the motion of the whole.
    The mechanical device is made from metal.
    The parts that make up the vehicle consist of the body, the frame, the suspension, the steering wheel, the brake, the engine, the wheels, the brakes, the transmission, the fuel tank, the fuel pump, the air compressor, and all other parts necessary to operate the vehicle.  The parts of the vehicle can be divided into two groups : mechanical and electrical.
    Mechanical parts
    The mechanical parts of the automobile consist of the wheels and the wheels themselves.  The mechanical parts of the automobile are called wheels and wheels.  A wheel is a unit of measurement that is made up of two parts that
    TOTAL TIME ELAPSED: 399.65s
    


```python
start = time.time()

#2ème génération
output = gpt2_lm_large.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  an automobile is a vehicle that is driven by a person.
    The definition of an automobile can be found in the US dictionary :
    Car (noun) : a vehicle designed for the purpose of transportation by human beings, including automobiles.
    Car (noun) : a vehicle designed for use by human beings.
    In the case of automobiles there are three parts of the vehicle that are considered to be the main components : The frame, the chassis and the wheels.
    The frame is the part of the automobile that contains the frame and wheels of a vehicle, and also the chassis that holds the vehicle together. It also contains the engine and transmission, and the tires that are used to move the vehicle.
    The frame is made of metal, and the parts that are made of metal are called the parts. The frame of an automobile is made of a steel plate that is made
    TOTAL TIME ELAPSED: 373.51s
    

#### Tests lié à divers sujets de la dataset scientific papers


```python
start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm_large.generate("How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

print("=======================================")

start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository pubmed
output = gpt2_lm_large.generate("Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    
    The structure of the multiplex can be a problem, as it can make it very difficult to understand the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the flow of the
    TOTAL TIME ELAPSED: 379.61s
    =======================================
    
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ? Coagulation factors, also known as coagulation factors A and B, are proteins that help to regulate cell activity and blood clotting. They help to keep blood clots in check, which helps to prevent bleeding and the spread of infections. PE is the name for an inherited disorder that is caused by mutations in the PE gene. This gene, also known as the Fibrinolysis protein, is located on chromosomes 21 and 23. The PE gene is a part of the gene family that codes for proteins. The PE gene is located on chromosome 21 and 23.
    
    The PE gene is located on chromosome 21 and 23 in people with PE. PE is the name for an inherited disease that is caused by mutations in the PE gene. This gene, also known as the Fibrinolysis protein, is located on chromosomes 21 and
    TOTAL TIME ELAPSED: 190.57s
    

## Tests de genération de texte sur GPT2 Extra Large
---
### Initialisation du modèle gpt2_extra_large_en


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_extra_large_en",
    sequence_length=128,
)

gpt2_lm_extra_large = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_extra_large_en", preprocessor=preprocessor
)
```

    Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large_en/v1/model.h5
    [1m6231301960/6231301960[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m141s[0m 0us/step
    

### Tests proposés sur keras.io


```python
start = time.time()

output = gpt2_lm_extra_large.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My trip to Yosemite was a great experience. It was my second time to Yosemite. I have always loved the place. The park itself is beautiful, but it was the surrounding mountains and the surrounding area that really got me. I love the fact that it is so close to the city. I have a few friends who have visited Yosemite and said that it is a must see.
    
    I am a huge fan of the park, I love the views, I love the scenery. I also love the people. I have been there a few times and it is just a really great place to be. The people are so friendly, and the food is really good.
    
    I was able to see a lot of the Yosemite area. It was really cool. I was lucky enough to see the Giant Sequoia and I was able to see Yosemite Falls, Yosemite Valley, Half Dome and El Capitan.
    
    I was lucky enough to get my picture taken with Half Dome. I
    TOTAL TIME ELAPSED: 457.43s
    


```python
start = time.time()

output = gpt2_lm_extra_large.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    That Italian restaurant is the first to be shuttered in the wake of the new rules.
    
    The new rules, which came into effect in January, are part of the EU's new migration policy, which has caused concern for some businesses, especially those with Italian owners and employees who are already struggling to cope with the economic fallout from the eurozone crisis.
    
    The rules, which were designed to help countries like Italy deal with the influx of migrants from North Africa and the Middle East, are designed to prevent people from claiming asylum in Europe and to make the journey to the European Union from Africa, Asia and the Middle East more difficult.
    
    The new rules require people arriving from Libya, Turkey and other countries to register with authorities before they are allowed to enter Europe.
    
    They also force migrants to stay in the first EU country they enter, rather than being allowed to apply for asylum in the first country where they enter.
    
    "It has been a very difficult year for the
    TOTAL TIME ELAPSED: 8.54s
    

### Tests personnels

#### Test complétion de texte


```python
start = time.time()

output = gpt2_lm_extra_large.generate("Working on this AI development homework,", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

```

    
    GPT-2 output:
    Working on this AI development homework, I've been thinking about the concept of "the human brain", and how the various parts interact. The idea is simple. Each part of the brain is a separate AI. They are not connected to each other, and they are not connected to the external world. Each part of the brain is a separate AI. Each part has its own goals, it has its own logic. Each part has its own memory. Each part has it's own personality, it's own personality. Each part is independent of all other parts of the brain. Each AI is self aware. They have a memory and they have goals. They have their own personality and they have their own goals.
    
    The AI is a separate entity. It has a memory, it has goals, and it is autonomous. It does not interact with the other AI, it doesn't interact with the external world, it is separate from the external world. It doesn't have a personality. It
    TOTAL TIME ELAPSED: 8.30s
    

#### Test traitement d'opérations mathématiques


```python
start = time.time()

output = gpt2_lm_extra_large.generate("1 + 1", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    1 + 1
    
    I was in the middle of an interview and my phone rang.
    
    "What's this?" my interviewer asked.
    
    "A friend of mine wants to see you," said the caller.
    
    "What's his name?"
    
    "David."
    
    "I'm sorry."
    
    "David. He wants to see you."
    
    The call ended there, and I had to call back later to ask for his number.
    
    I'm a little bit of an introvert, and when I first met David, he was the kind of person who would be very hard to approach. He was quiet and reserved. And yet, he was also incredibly charismatic.
    
    When I finally did approach him, David told me he was a big football fan. He was also an expert in the field of business, and he had recently started his own business.
    
    We had a long conversation and then I told him how much I admired
    TOTAL TIME ELAPSED: 7.71s
    

#### Test question générale


```python
start = time.time()

output = gpt2_lm_extra_large.generate("My question is : what is an automobile consisted of ? My answer to the previous question is : ", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    My question is : what is an automobile consisted of ? My answer to the previous question is :  a vehicle, a vehicle that carries people, or a vehicle that is used for transportation. I don't really have an answer for your question. I think that it is a vehicle that is capable of being used for transportation and that is capable of carrying people.
    What is an automobile ?
    An automobile is a vehicle that is designed and built for the transportation of people. The term automobile was first used in the United States in 1869 to describe a steam-powered vehicle.
    The term automobile was used by the American automobile industry to describe a car, a motor vehicle, or a truck.
    An automotive vehicle can be used for any number of purposes, including transportation and for personal use. The term automobile is also used to describe any vehicle with wheels. In this context, the term is used to describe a vehicle with wheels, regardless of whether or not the vehicle has wheels.
    TOTAL TIME ELAPSED: 739.26s
    

#### Tests lié à divers sujets de la dataset scientific papers


```python
start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository arxiv
output = gpt2_lm_extra_large.generate("How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

print("=======================================")

start = time.time()

# Extrait d'un article scientifique de la dataset tensorflow scientific papers, repository pubmed
output = gpt2_lm_extra_large.generate("Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    
    In this article we'll try to answer this question. We'll use a simple multiplex with two gates and a passenger waiting area.
    
    The multiplex consists of two gates, a waiting area and a passenger waiting area.
    
    We'll use the same example as in the previous article about the structure of an application to the structure of a multiplex.
    
    The main difference is that the multiplex consists of two gates, a waiting area and a passenger waiting area.
    
    Let's see the main differences.
    
    The multiplex consists of two gates, two waiting areas and a passenger waiting area.
    
    The multiplex consists of two gates, two waiting areas and a passenger waiting area.
    
    The main difference is that the multiplex consists of two gates, two waiting areas and a passenger waiting area.
    
    The main difference is that the multiple
    TOTAL TIME ELAPSED: 444.83s
    =======================================
    
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    
    Answer: Coagulation factor IX (CXC9) is the most common inherited coagulation defect. The defect results from mutations in the gene for CXC9. In the majority of individuals, the gene is on chromosome 17. However, in some individuals with CXC9 defects, the gene is on chromosome 18, which is known as the CXC9 18p deletion.
    
    PE is the Inherited defect of fibrinolysis. The defect results from mutations in the gene for Fibrinogen. The gene for Fibrinogen is on chromosome 15. In some individuals with PEs, the gene for Fibrinogen is on chromosome 16, which is known as the Fibrinogen 16p deletion.
    
    Consequently, in most individuals, the CXC
    TOTAL TIME ELAPSED: 421.99s
    

## Observations suite aux tests sur les modèles gpt2 medium, large et extra large
---
En effectuant les mêmes tests réalisés avec GPT2 Base sur les modèles Medium, Large et Extra large contenant un nombre plus élevé de couches et de paramètres, on constate qu'on obtient dans certains cas des résultats plus précis et cohérent.

Cela est le cas pour l'exemple de questionnement sur la constitution d'une automobile, où le modèle de base à 12 couches répondait "Je ne sais pas" ou "Cela consiste à un moteur et c'est une automobile", alors que le modèle large et extra large donnaient des réponses plus précises, fournissant la définition du terme, et des termes techniques sur les voitures : suspension, steering wheel, brakes.

GPT-2 parvient donc à approfondir l'utilisation des termes techniques en rapport avec le sujet abordé, d'un manière de mieux saisir les enjeux de la saisie, de la question.

Néanmoins, le modèle pouvant accéder et traiter un plus large champs de données, il est susceptible de fournir des informations fausses. Notamment pour le terme "automobile" selon GPT2 utilisé pour la première fois aux Etats-Unis en 1869, alors qu'en réalité l'année correcte est en 1885. Cette problématique concerne aussi le modèle de base gpt2_base_en mais à un niveau différent, et ces tests montrent donc que cela est une problématique générale du modèle GPT-2.

Il peut être noté que l'utilisation de ces modèles comportant plus de couches et de paramètres, nécessite tout de même beaucoup plus de temps et de ressources pour générer une réponse plus élaborée, plus réfléchie.

Le modèle GPT2 reste donc un modèle très focalisé sur la génération et complétion de texte, et est moins adapté à un format question-réponse, voir donc à une utilisation en tant que chatbot. Cela reste pour autant possible et réalisable.


```python

```
