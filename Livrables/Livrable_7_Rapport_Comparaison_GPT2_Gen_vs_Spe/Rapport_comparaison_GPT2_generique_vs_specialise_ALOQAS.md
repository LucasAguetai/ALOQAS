# Rapport sur la comparaison entre le modèle générique et le modèle spécialisé

Dans ce notebook, nous effectuons une comparaison entre le modèle GPT-2 générique et le modèle GPT-2 spécialisé, en comparant la génération de texte sur des prompts techniques spécifiques, et en effectuant des tests d'évaluation en se basant sur le jeu de données d'articles scientifiques.

L'objectif est de pouvoir d'observer la différence en termes de qualité et de précision des réponse du modèle par rapport à des questions/prompts techniques et scientifiques, entre le modèle générique et spécialisé.

**Groupe ALOQAS**

- Aurélien ZUFIC
- Lucas AGUETAÏ
- Ony ANDRIATSAHAVOJAONA
- Quentin VERMEERSCH
- Alexandre HUYNH
- Samuel DORISMOND


---
## Import de packages


```python
!pip install keras_nlp -q
!pip install datasets -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m415.4/415.4 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m950.8/950.8 kB[0m [31m15.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m536.6/536.6 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.3/38.3 MB[0m [31m20.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m16.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m17.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    ibis-framework 7.1.0 requires pyarrow<15,>=2, but you have pyarrow 15.0.0 which is incompatible.[0m[31m
    [0m


```python
import tensorflow as tf
from google.colab import drive
import json
import os
import sys
import keras_nlp
import time
from tensorflow import keras
import numpy as np

drive.mount('/content/drive', force_remount=True)
sys.path.append('/content/drive/MyDrive/package')
#import datasets_scientific_paper as load_ds
```

    Using TensorFlow backend
    Mounted at /content/drive
    


```python
from datasets import load_dataset

dataset = load_dataset("scientific_papers", "pubmed")
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/datasets/load.py:1454: FutureWarning: The repository for scientific_papers contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/scientific_papers
    You can avoid this message in future by passing the argument `trust_remote_code=True`.
    Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
      warnings.warn(
    


    Downloading builder script:   0%|          | 0.00/5.35k [00:00<?, ?B/s]



    Downloading readme:   0%|          | 0.00/8.27k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/3.62G [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/880M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/119924 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/6633 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/6658 [00:00<?, ? examples/s]


---
## Initialisation du modèle GPT-2


```python
os.environ["KERAS_BACKEND"] = "tensorflow"

keras.mixed_precision.set_global_policy("mixed_float16")

preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)
```

    Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/tokenizer.json...
    100%|██████████| 448/448 [00:00<00:00, 299kB/s]
    Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/assets/tokenizer/merges.txt...
    100%|██████████| 446k/446k [00:00<00:00, 2.94MB/s]
    Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/assets/tokenizer/vocabulary.json...
    100%|██████████| 0.99M/0.99M [00:00<00:00, 4.84MB/s]
    Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/config.json...
    100%|██████████| 484/484 [00:00<00:00, 488kB/s]
    Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/model.weights.h5...
    100%|██████████| 475M/475M [00:05<00:00, 86.5MB/s]
    /usr/local/lib/python3.10/dist-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
      return id(getattr(self, attr)) not in self._functional_layer_ids
    /usr/local/lib/python3.10/dist-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `layer.updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
      return id(getattr(self, attr)) not in self._functional_layer_ids
    

---
## Génération de texte avec le modèle GPT-2 générique (sans fine-tuning)


```python
# Fonction génération de texte
def generate_text(prompt):
    start = time.time()
    output = gpt2_lm.generate(prompt, max_length=200)
    end = time.time()
    print('________________________________________________')
    print(f"\nPrompt: {prompt}")
    print("GPT-2 output:")
    print(output)
    print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
    #return output
```


```python
# Prompts génération de texte
prompts = [
    "Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that", # extrait article repository arxiv
    "How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?", # extrait article repository arxiv
    "Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?", # extrait article repository pubmed
    "What are the latest advancements in cancer research ?", # exemple suggestion personnelle
    "What is the impact of diet on heart disease according to recent studies ?", # exemple suggestion personnelle
    "What are the usual causes of lung pain ?", # exemple suggestion personnelle
    "", # génération vide
    "" # génération vide
]
```


```python
# Génération de texte avec le modèle générique

print("Output GPT-2 générique")
i=0
while i < len(prompts):
  generate_text(prompts[i])
  i+=1
```

    Output GPT-2 générique
    ________________________________________________
    
    Prompt: Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that
    GPT-2 output:
    Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that the first step in the process of hypertrophy is the removal of the sympathetic nervous system from the heart, and this process involves the removal of the blood-brain barrier (BBB), the main source of the blood supply to the central nervous system. The BBB is a central component of the heart and, therefore, is the primary organ of circulation for the heart.
    
    The heart is a large body of blood, with many different types, and the BBB, in particular the blood supply to the heart, is a major organ of the central nervous system (CNS), which is responsible for the flow of blood to the heart. The blood supply to the heart is regulated by several factors, including:
    
    Cardiovascular disease (heart failure)
    
    Cardiovascular diseases (stroke, heart failure)
    
    Heart failure
    TOTAL TIME ELAPSED: 43.35s
    ________________________________________________
    
    Prompt: How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ? The answer is in the form of a network that connects all passengers to the same airline. The network is a multi-tier, multi-level structure, and can be used to create a multi-level network that is not only flexible, but is scalable. The multi-tier structure allows a network that is multi-tier to connect to a wide variety of airlines in a single place, including: United, Air France, Air France-KLM, and Air France-CAC, and also to connect to many other airlines in a single place, from United to United to Air Canada.
    
    The multi-tier structure allows a network that is multi-tier to connect to a wide range of airlines in a single place, including: United, Air France, Air France-KLM, and Air France-CAC, and also to connect to many other airlines in a single place
    TOTAL TIME ELAPSED: 26.04s
    ________________________________________________
    
    Prompt: Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    
    It is not the Inherited defect of fibrinolysis but the Inherited defect of PE Inherited defects. In coagulation factors, the Inherited defect is the same, but the Inherited defect is a little different in that it can be inherited in coagulation factors only. It is also possible that the inherited defect has some effect on coagulation factor.
    
    The Inherited defect of coagulation factors is that coagulation factor is not present at the moment. But it may be present in coagulation factors, and the Inherited defect is present in coagulation factors. It is not clear why the Inherited defect is different in coagulation factors.
    
    The Inherited defect of coagulation factors may be inherited from a coagulation factor that is different from the coag
    TOTAL TIME ELAPSED: 24.99s
    ________________________________________________
    
    Prompt: What are the latest advancements in cancer research ?
    GPT-2 output:
    What are the latest advancements in cancer research ? The latest advancements in cancer research
    
    In the United States, the cancer rate for women aged 15 to 64 is on a downward trend, from 7 percent in 1999 to 5.5 percent in 2014. The number of cancer cases is also declining. The number of cancer patients in the United States is on a downward trend, from 1.3 percent in 1999 to 5.1 percent in 2014. The cancer rate for women aged 15 to 64 is on a downward trend , from 1.4 percent in 1999 to 5.3 percent in 2014. The number of cancer patients in the United States is on a downward trend , from 1.4 percent in 1999 to 5.3 percent in 2014.
    
    In 2014, a total of 1,739 women aged 15 to 64 (1.7%) were diagnosed with breast cancer, compared to 1,829 (1.8%) who had never been diagnosed with breast cancer, according to
    TOTAL TIME ELAPSED: 28.19s
    ________________________________________________
    
    Prompt: What is the impact of diet on heart disease according to recent studies ?
    GPT-2 output:
    What is the impact of diet on heart disease according to recent studies ?
    
    
    The most recent research on the impact of diet on cardiovascular disease has shown that the risk of heart disease is reduced with increasing amounts of saturated fat, and that saturated fats are associated with increased risk of heart disease. The risk reduction from dietary fats is associated with a reduction in risk of heart disease in the elderly and in the obese. In fact, the risk reduction from diet has been associated with a reduction in the risk of heart disease among the obese. The most recent evidence for a reduction from saturated fats in heart disease has shown that the risk of heart disease is reduced with increasing amounts of saturated fat, and that saturated fats are associated with increased risk of heart disease.
    
    
    The most recent research on the effect of diet on cardiovascular disease has shown that the risk of heart disease is reduced with increasing amounts of saturated fat, and that saturated fats are associated with increased risk of heart disease. In fact, the risk reduction
    TOTAL TIME ELAPSED: 26.21s
    ________________________________________________
    
    Prompt: What are the usual causes of lung pain ?
    GPT-2 output:
    What are the usual causes of lung pain ?
    
    A lot. The first thing to look out for is that you are not breathing. If you are breathing, there are a few common causes of lung pain:
    
    Lungs
    
    Lungs are a common cause of lung pain. The cause of lung pain can vary by person. In general, the more you are breathing, the more likely you are to have lung pain.
    
    Lungs may be a common cause of lung pain. The cause of lung pain can vary by person. In general, the more you are breathing, the less likely you are to have lung pain. Sudden death syndrome
    
    The condition that causes lung pain is called sudden death syndrome (SD). This is a common cause of death for those with asthma. The reason that people are more likely to have SD is because they have a higher rate of asthma, so they have more of a risk of lung disease.
    
    What
    TOTAL TIME ELAPSED: 27.81s
    




    '\nprint(output_generique0 + "\n")\nprint(output_generique1 + "\n")\nprint(output_generique2 + "\n")\n'




```python
print("Output GPT-2 générique - Prompt string vide")
generate_text("")
generate_text("")
```

    Output GPT-2 générique - Prompt string vide
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    "I don't think you should be able to go on the Internet, and that's why you're doing it. It's not a good idea," he said.
    
    The former Republican senator from Texas was among a handful of lawmakers who have taken to Twitter to express their support for Mr. Trump's travel ban.
    
    "The people of the U.S. have a right to know who the president is and what his intentions are," said Sen. John McCain, R-Ariz., who was among those who took to the platform to call for "an independent and impartial investigation" of the ban.
    
    The president's executive order temporarily blocked immigration from seven Muslim-majority countries, including Iran, Iraq, Libya, Somalia, Sudan, Syria and Yemen.
    
    The ban is a response to a series of high-profile incidents in recent years that have raised questions about whether President Trump's executive actions will have a negative effect on U.S. security. The
    TOTAL TIME ELAPSED: 18.61s
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    "We've seen a lot in the last couple of months," he said, "we've seen a lot of people who want to get involved and we've seen a lot of people who think it's not going to happen."
    
    "The only thing I can say for sure right now is that this is the best we have seen in terms of the way it's been handled."
    
    The mayor, who was in the audience, said there was a lot that could happen if he were elected.
    
    "We've got to be realistic and make sure that this is what happens. This is not going to happen. I think we've seen a lot of people who want to do this," he said. "We're going to have a very difficult time."
    
    The mayor said he was pleased the community was taking the time to hear from the community.
    
    "I think it's really nice to have a community like that," he said, "and that's
    TOTAL TIME ELAPSED: 0.98s
    

---
## Test d'évaluation du modèle GPT-2 générique (sans fine-tuning)


```python
val_features = dataset["validation"][:]["article"]

# Évaluer le modèle sur les données de validation
with tf.device('/device:GPU:0'):
  val_loss, val_accuracy = gpt2_lm.evaluate(
      x=val_features,
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 générique - Validation Loss: {val_loss}")
print(f"GPT-2 générique - Validation Accuracy: {val_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    208/208 [==============================] - 3346s 16s/step - loss: 3.9689 - sparse_categorical_accuracy: 0.3391
    Validation Loss: 3.9688880443573
    Validation Accuracy: 0.33905521035194397
    


```python
test_features = dataset["test"][:]["article"]

# Évaluer le modèle sur les données de test
with tf.device('/device:GPU:0'):
  test_loss, test_accuracy = gpt2_lm.evaluate(
      x=np.array(test_features),
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 générique - Test Loss: {test_loss}")
print(f"GPT-2 générique - Test Accuracy: {test_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    209/209 [==============================] - 3310s 16s/step - loss: 3.9504 - sparse_categorical_accuracy: 0.3406
    GPT-2 spécialisé - Test Loss: 3.9503707885742188
    GPT-2 spécialisé - Test Accuracy: 0.34061935544013977
    

---
## Génération de texte avec le modèle GPT-2 spécialisé (avec fine-tuning)

Finetuné sur le jeu de données d'articles scientifiques.


```python
# Chargement des paramètres GPT-2 finetuné sur les articles scientifiques
#checkpoint_path = "/content/drive/MyDrive/training_data_all_remake/cp.ckpt"
checkpoint_path = "/content/drive/MyDrive/training_data_all_remake_base_epochs/cp.ckpt"

gpt2_lm.load_weights(checkpoint_path)
```




    <tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus at 0x7ecbf33ba110>



### Modèle GPT-2 finetuné - 1ère version

GPT-2 Base, entrainé sur dataset téléchargé et divisé manuellement


```python
# Génération de texte avec le modèle spécialisé

print("Output GPT-2 spécialisé")
i = 0
while i < len(prompts):
  generate_text(prompts[i])
  i+=1
```

    Output GPT-2 spécialisé
    ________________________________________________
    
    Prompt: Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that
    GPT-2 output:
    Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that there is a growing body of evidence indicating the association between cardiometabolic changes and cardiovascular disease . the relationship between cardiometabolic and cardiovascular risk is well established.2 the role of the heart is to support the development of a cardiometabolic state , and the role of the heart is to support the development of a cardiometabolic state . the heart is a complex organ and its function requires the ability to maintain and regenerate its integrity , and the heart is the
    TOTAL TIME ELAPSED: 32.71s
    ________________________________________________
    
    Prompt: How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ? it is a well known fact that the structure of a network is not the only one that determines the success of the flight ( e.g. , [ 1 , 2 ] ) . for example , the shape of the network is influenced by the shape of the transport network , which in turn is determined by the location of the transport system . the shape of a network depends on the location and orientation of the transport system . the shape of a network depends on the shape of the transport system and the position of the transport system . the shape
    TOTAL TIME ELAPSED: 15.57s
    ________________________________________________
    
    Prompt: Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ? 
      my question is simple : what is the inherited defect of fibrinolysis , which is caused by a defective production of a coagulation factor in the absence of a defect of the coagulation factors , which is produced by the defective production of an fibrinolysis in a deficient fashion ? 
      my question is simple : what is the inherited defect of fibrinolysis , which is caused by a defective production of a coagulation factor in
    TOTAL TIME ELAPSED: 14.74s
    ________________________________________________
    
    Prompt: What are the latest advancements in cancer research ?
    GPT-2 output:
    What are the latest advancements in cancer research ? it is not only a matter of time for the development of new therapies , but also for the establishment of new technologies . in the last decade , cancer has been a major public health problem . the incidence rate of cancer in the us is increasing , and the incidence rate of cancer is expected to increase in the next decade . in the us , there are two types of cancer : solid cancers and non - solid cancers ( nc ) ( 1 ) . the most solid cancers , which are considered the second most prevalent cancer , are solid tumors and the second most common cause of cancer mortality . however
    TOTAL TIME ELAPSED: 17.31s
    ________________________________________________
    
    Prompt: What is the impact of diet on heart disease according to recent studies ?
    GPT-2 output:
    What is the impact of diet on heart disease according to recent studies ? the main role of dietary supplementation has been the treatment of choice for patients with coronary heart disease ( chd ) . however , a large number of patients have been treated with dietary supplements , which may be harmful for cardiovascular and renal disease . the most common adverse reactions associated with dietary supplementation are cardiovascular ( cvd ) symptoms such as hypertension ( htn ) , diabetes , and hypertension ( htn ) , and renal ( rtn ) , but the effect on renal function is still controversial . several studies have been conducted to find a relationship between diet and cardiovascular ( cvd
    TOTAL TIME ELAPSED: 16.30s
    ________________________________________________
    
    Prompt: What are the usual causes of lung pain ?
    GPT-2 output:
    What are the usual causes of lung pain ? in the literature , the causes are mostly attributed to the inflammation of the bronchi , which is caused by bronchial epithelium . the most frequently reported causes of pain are bronchi , bronchial fibroma , and bronchi and pulmonary fibroma . a large number of studies have reported that chronic bronchial inflammation is the most important factor in lung pain . in addition , there is a lack of data about the causes of lung pain . the aim of our study was to compare the results of two studies in patients with chronic lung pain . we included patients aged 40 - 70
    TOTAL TIME ELAPSED: 17.57s
    




    '\nprint(output_specialise0 + "\n")\nprint(output_specialise1 + "\n")\nprint(output_specialise2 + "\n")\n'



### Modèle GPT-2 finetuné - 2ème version

GPT-2 base, entrainé sur dataset importé de Hugging Face, identique à celle de Tensor Flow


```python
# Génération de texte avec le modèle spécialisé

print("Output GPT-2 spécialisé")
i = 0
while i < len(prompts):
  generate_text(prompts[i])
  i+=1
```

    Output GPT-2 spécialisé
    ________________________________________________
    
    Prompt: Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that
    GPT-2 output:
    Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that a number of studies suggest that hypertrophy of adipose tissue ( at ) is the cause of at , and that this is associated with increased risk of coronary events .
    these findings are supported by the evidence that the metabolic rate ( mr ) is elevated during the first 2 weeks of atherosclerosis and that the mr is elevated in patients with coronary artery disease ( cad ) and that the mr is elevated in patients with non - cad [ 2 , 3 ] .
    TOTAL TIME ELAPSED: 1.73s
    ________________________________________________
    
    Prompt: How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    one of the main reasons for the lack of a walkway is the lack of a walkway that is capable of moving and moving at a precise speed .
    the main reason is the fact that the walkway needs an additional arm of a dedicated wheelchair , which may lead the user to a greater risk of falling down .
    the other reason for a walkway to be capable of moving at a precise speed is the fact that the walkway is designed to allow a person to use the wheelchair for a prolonged period of time and to
    TOTAL TIME ELAPSED: 1.58s
    ________________________________________________
    
    Prompt: Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    a 32-year - old man presented with a 2-year history of a painless swelling over right lower extremity of the left knee since 3 years .
    the swelling had been growing slowly over the previous 3 years , and the patient had been on anti - fibrinolytic medication since the last 2 years .
    he had been on anti - fibrinolytic medications for the last 2 years . on admission ,
    the lesion was gradually enlarging over the right
    TOTAL TIME ELAPSED: 2.13s
    ________________________________________________
    
    Prompt: What are the latest advancements in cancer research ?
    GPT-2 output:
    What are the latest advancements in cancer research ? 
      the world health organization ( who ) defines cancer as the 
                                                                                                          
    TOTAL TIME ELAPSED: 0.64s
    ________________________________________________
    
    Prompt: What is the impact of diet on heart disease according to recent studies ?
    GPT-2 output:
    What is the impact of diet on heart disease according to recent studies ?
    the study of the prevalence and severity of type 2 diabetes mellitus and cardiovascular disease in children and adolescents in the uk has been a long and well - conducted study .
    the study of the prevalence of type 2 diabetes mellitus and cardiovascular disease in adolescents has been a long and well - conducted study . the study has been reported to have a high prevalence of cardiovascular events , hypertension , diabetes , dyslipidemia , hyperlipidemia , and dyslipidemia in children and adolescents .
    it has been suggested that type 2 diabetes mellitus and cardiovascular
    TOTAL TIME ELAPSED: 0.63s
    ________________________________________________
    
    Prompt: What are the usual causes of lung pain ?
    GPT-2 output:
    What are the usual causes of lung pain ?     
    it is generally accepted that the main causes are respiratory infection and the respiratory 
     response to the inhalation of aerosols .
    however , in many cases , the main cause is inhalation of aerosols 
     by inhalation of aerosols by the respiratory response of the respiratory 
     respiratory tract .
    the respiratory response to inhalation of aerosols causes respiratory pain , and the 
     respiratory response is the main reason for the lung pain .
    the respiratory response to inhalation of aerosoids is 
     mainly determined by the respiratory reflex of the
    TOTAL TIME ELAPSED: 1.00s
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    the term  noninvasive  refers to non - invasive , non - invasive , non - surgical methods for the treatment of patients who are at high risk for infection .
    noninvasive methods include the use of an invasive approach to treat infection , and the use of a noninvasive approach to treat infection .
    noninvasive methods include surgical procedures , including surgical resection , radiotherapy , and endoscopic surgery , and noninvasive procedures such as the surgical removal of the spleen and liver .
    the use of noninvasive procedures in the management of infection is a common indication for the use of invasive techniques .
    
    TOTAL TIME ELAPSED: 0.55s
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    the study was approved by the ethics committee of china ministry of education and science ( mcmh , no .
    the study was conducted in accordance with the guidelines of the declaration of helsinki ( 2 ) .
    subjects were selected by the age of 18 - 40 years and the mean age of subjects was 25.4 years .
    the subjects were divided in two stages : a ) the first group consisted of 2 groups of children with a normal weight , height , and weight distribution ; b ) a second group consisted of 2 groups of children with a normal body shape ; and c ) 2 groups of children with normal body
    TOTAL TIME ELAPSED: 0.56s
    

**Observations**

Nous observons certains défauts de génération, tels que lors de la génération avec le prompt "What are the latest advancements in cancer research ?" ayant une phrase non complétée.

Le prompt "What are the usual causes of lung pain ?" semble bien répondre à la question en citant une des causes de douleurs pulmonaires, même si des répétitions peuvent être soulignés.

Un constat peut être fait concernant les prompts vides, où le texte généré est très orienté vers ceux du jeu de données, pouvant s'apparenter quasimment à un extrait d'articles scientifiques.

### Modèle GPT-2 finetuné - 3ème version

GPT-2 base, entrainé en 2 fois (50/50% articles train)sur dataset importé de Hugging Face, avec 3 epochs pour chaque parties entrainés.


```python
# Génération de texte avec le modèle spécialisé

print("Output GPT-2 spécialisé")
i = 0
while i < len(prompts):
  generate_text(prompts[i])
  i+=1
```

    Output GPT-2 spécialisé
    ________________________________________________
    
    Prompt: Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that
    GPT-2 output:
    Cardiac hypertrophy is initiated as an adaptive response to sustained overload but progresses pathologically as heart failure ensues. On this topic, I know that the most common cause of heart failure is myocardial infarction ( mi) .
    it occurs when a large number of myocardial cells die in the event of a cardiac event , and the loss of cells is accompanied by an increase in intracellular calcium concentration .
    this is a major complication of mi , with the incidence of myocardial infarction ( mi ) increasing with advancing age . in older age , the incidence of mi
    is about 10 times
    TOTAL TIME ELAPSED: 0.58s
    ________________________________________________
    
    Prompt: How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    GPT-2 output:
    How does the structure of a multiplex affects the walk applications to real-world airline transportation networks ?
    a large number of studies have examined the effect of multiplex on walking performance and have found that multiplex has a negative impact on the ability to perform daily activities .
    many studies have shown that multiplex is an important factor in the ability of passengers to walk safely .
    the effects of multiplex on walking have been studied extensively , with many reports indicating that multiplex has a negative effect on ability to walk , particularly for older persons with disabilities . in a recent study ,
    the effects of multiplex on walking ability were evaluated
    TOTAL TIME ELAPSED: 0.78s
    ________________________________________________
    
    Prompt: Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    GPT-2 output:
    Could you explain to me what is the Inherited defect of coagulation factors and PE Inherited defects of fibrinolysis ?
    it is an important issue for the medical field , because it is the most important cause of mortality and the most common cause of cardiovascular morbidity in the world .
    the coagulative factor is a protein that plays a central role in the regulation of the vascularity .
    it is a member of a family of proteins known as the interleukin-2 receptor superfamily .
    it is expressed at the level of the endothelium and is activated at the level of other cells
    TOTAL TIME ELAPSED: 0.51s
    ________________________________________________
    
    Prompt: What are the latest advancements in cancer research ?
    GPT-2 output:
    What are the latest advancements in cancer research ? the number of cancer patients is on the rise in the world .
    there are many new cancers diagnosed in india and in the usa .
    the most recent report of the international cancer society ( icus ) on cancer in india ( 2008 ) reported that there is a 7.5% increase in the number of cancer cases in 2008 . in india
    , it is well known that the incidence of cancer in females is higher , but it has been known that in males , it is lower .
    this has been attributed to differences in tumor stage , tumor stage , and cancer
    TOTAL TIME ELAPSED: 0.54s
    ________________________________________________
    
    Prompt: What is the impact of diet on heart disease according to recent studies ?
    GPT-2 output:
    What is the impact of diet on heart disease according to recent studies ? 
    
    there is evidence that dietary intake of saturated fat and low - saturated fatty acids ( lfas ) 
     increases the risk of coronary heart disease in patients with type 2 diabetes .
    furthermore , dietary 
     fat and low - saturated fatty acids increase the risk of coronary heart disease in patients with type  2 diabetes .
    thus , 
     there is growing interest in the role of dietary fat and lfas in the prevention , prevention , and treatment 
     of coronary heart diseases .
    however , the mechanisms by which dietary factors
    TOTAL TIME ELAPSED: 0.80s
    ________________________________________________
    
    Prompt: What are the usual causes of lung pain ?
    GPT-2 output:
    What are the usual causes of lung pain ?
    the most common cause of lung pain is bronchial asthma , with a lifetime prevalence of 2% .
    it has been known since the early 1970s that the lung has the ability to respond to environmental stimuli and to maintain its own homeostatic function by controlling the release of inflammatory mediators and other inflammatory mediators .
    however , the exact mechanisms by which the lung responds to environmental stimuli are still not fully understood , and it has been shown that some environmental factors , including environmental factors and cigarette smoking , can increase the risk of lung disease .
    the most common environmental factor
    TOTAL TIME ELAPSED: 0.53s
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    this study was a prospective observational , multicenter , observational study of the effects of a single oral antithrombin inhibitor , nivolumab , on the clinical course of hcc in patients with type 1 diabetes mellitus ( t1 dm ) .
    patients with diabetes were recruited from the general practice of the university of michigan medical center , which provides primary care to patients in diabetes care ( www.ucm.ca ) and other diabetes care ( www.diabetes-council.ca ) .
    patients were excluded if they met criteria for hcc , were not taking any medications or
    TOTAL TIME ELAPSED: 0.54s
    ________________________________________________
    
    Prompt: 
    GPT-2 output:
    the study was approved by the ethics committees of the university of korea , and the institutional review board of the university hospital korea ( koh ) .
    the study was approved by the ethics committee of the university hospital koh .
    the study was conducted on a cohort of patients with type 1 diabetes ( n = 1,964 ) who had undergone endoscopic retrograde cholangiopancreatography ( ercp ) for the past 6 months .
    all patients underwent ercp with either an endoscopic retrograde cholangiopancreatography ( ercp - e ) or endoscopic retrograde cholang
    TOTAL TIME ELAPSED: 0.55s
    

**Observations**

Nous observons ici que les résultats sont nettement meilleurs par rapport aux autres modèles.

Avec le prompt "What is the impact of diet on heart disease according to recent studies ?", le modèle évoque bien certaines causes précises avec des termes techniques tels que "saturated fat and low - saturated fatty acids ( lfas )"

Nous avons même un cas de génération ressemblant beaucoup à une vraie interaction question-réponse avec "What are the usual causes of lung pain ?", donnant comme réponse "
the most common cause of lung pain is bronchial asthma , with a lifetime prevalence of 2%" (Quelles sont les causes des douleurs pulmonaires ? -> La cause la plus commune de douleur pulmonaire est...).

---
## Test d'évaluation du modèle GPT-2 spécialisé (avec fine-tuning)

Finetuné sur le jeu de données d'articles scientifiques.

### Modèle GPT-2 finetuné - 2ème version

GPT-2 base, entrainé sur dataset importé de Hugging Face, identique à celle de Tensor Flow


```python
val_features = dataset["validation"][:]["article"]

# Évaluer le modèle sur les données de validation
with tf.device('/device:GPU:0'):
  val_loss, val_accuracy = gpt2_lm.evaluate(
      x=val_features,
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 spécialisé - Validation Loss: {val_loss}")
print(f"GPT-2 spécialisé - Validation Accuracy: {val_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    208/208 [==============================] - 209s 939ms/step - loss: 2.8854 - sparse_categorical_accuracy: 0.4459
    GPT-2 spécialisé - Validation Loss: 2.885439872741699
    GPT-2 spécialisé - Validation Accuracy: 0.44592615962028503
    


```python
test_features = dataset["test"][:]["article"]

# Évaluer le modèle sur les données de test
with tf.device('/device:GPU:0'):
  test_loss, test_accuracy = gpt2_lm.evaluate(
      x=test_features,
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 spécialisé - Test Loss: {test_loss}")
print(f"GPT-2 spécialisé - Test Accuracy: {test_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    209/209 [==============================] - 207s 977ms/step - loss: 2.8649 - sparse_categorical_accuracy: 0.4483
    GPT-2 spécialisé - Test Loss: 2.86492657661438
    GPT-2 spécialisé - Test Accuracy: 0.44825422763824463
    

### Modèle GPT-2 finetuné - 3ème version

GPT-2 base, entrainé en 2 fois (50/50% articles train)sur dataset importé de Hugging Face, avec 3 epochs pour chaque parties entrainés.


```python
val_features = dataset["validation"][:]["article"]

# Évaluer le modèle sur les données de validation
with tf.device('/device:GPU:0'):
  val_loss, val_accuracy = gpt2_lm.evaluate(
      x=val_features,
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 spécialisé - Validation Loss: {val_loss}")
print(f"GPT-2 spécialisé - Validation Accuracy: {val_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    208/208 [==============================] - 201s 949ms/step - loss: 2.7834 - sparse_categorical_accuracy: 0.4607
    GPT-2 spécialisé - Validation Loss: 2.783418655395508
    GPT-2 spécialisé - Validation Accuracy: 0.46072784066200256
    


```python
test_features = dataset["test"][:]["article"]

# Évaluer le modèle sur les données de test
with tf.device('/device:GPU:0'):
  test_loss, test_accuracy = gpt2_lm.evaluate(
      x=test_features,
      verbose=1
  )

# Afficher les résultats de l'évaluation
print(f"GPT-2 spécialisé - Test Loss: {test_loss}")
print(f"GPT-2 spécialisé - Test Accuracy: {test_accuracy}")
```

    WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning.
    

    209/209 [==============================] - 217s 1s/step - loss: 2.7643 - sparse_categorical_accuracy: 0.4631
    GPT-2 spécialisé - Test Loss: 2.764336347579956
    GPT-2 spécialisé - Test Accuracy: 0.4630672335624695
    

---
## Observations et remarques concernant la génération de texte

On peut remarquer que les réponses du modèle GPT-2 Générique sont vagues, répétitives, et manquent de clarté. Elle ne fournit pas d'informations utiles ou précises sur le sujet. Alors que les réponses modèle GPT-2 Spécialisé sont légèrement plus structurées et semblent se concentrer davantage sur le sujet, bien qu'elle reste encore vague et manque de détails spécifiques.

Sur la question des avancées récentes sur le cancer, le générique se concentre sur des statistiques générales plutôt que sur les avancées spécifiques dans la recherche sur le cancer. Elle manque de détails pertinents sur le sujet, alors que le spécialisé mentionne l'évolution des thérapies et des technologies, ce qui est plus pertinent, mais elle ne fournit pas d'exemples concrets ou de détails spécifiques sur les avancées.

Sur la question de l'impact de l'alimentation sur les maladies cardiaques, la réponse du générique est contradictoire et peut induire en erreur, en particulier en ce qui concerne les graisses saturées. Alors que la réponse du spécialiste aborde les compléments alimentaires et leurs effets sur les maladies cardiaques de manière plus nuancée, bien qu'elle ne soit pas entièrement claire ou concluante.

Sur la question des causes habituelles de la douleur pulmonaire, le générique fournit des informations incorrectes et non pertinentes, telles que la mention du syndrome de mort subite. Le spécialisé se concentre sur des causes plus plausibles telles que l'inflammation bronchique, bien qu'il manque de profondeur dans l'explication.

Nous constatons en particulier qu'une génération de texte effectué sur un prompt vide via le modèle spécialisé oriente bien le sujet vers une thématique scientifique / médicale, démontrant que le modèle a bien pris en compte l'entrainement par fine-tuning.

En se basant sur les réponses obtenus pour le modèle spécialisé, nous constatons que les réponses

Certains défauts peuvent être relevés, tels que l'absence de majuscule en début de phrase ou encore la présence d'espaces avant les signes de ponctuation, rendant les réponses moins lisibles.

On peut donc en conclure que le modèle GPT-2 spécialisé semble offrir des réponses légèrement plus structurées et centrées sur les sujet traités dans les articles scientifques que celui générique, avec des termes scientifiques et des explications plus techniques.<br /> Cependant, même dans le cas du modèle spécialisé, les réponses manquent souvent de détails spécifiques et cohérents, de précision et de clarté.<br />
Nous pouvons tout de même souligner le fait que les réponses deviennent de plus en plus cohérentes et précises au fur et à mesure que nous optimisons le modèle, et paraissent réellement comme des vrais réponses aux questions. Cela est surtout observable avec le modèle fine-tuné avec GPT-2 base avec 3 epochs, où les réponses sont plus précises, cohérentes et bien construites.

---
## Observations et remarques concernant les tests d'évaluation des modèles

En effectuant des tests d'évaluation avec le jeu de données d'articles scientifiques, spécifiquement les parties d'évaluation et de test, nous avons pu constaté une augmentation de précision (accuracy) et une diminution en erreurs (loss) entre le modèle GPT-2 par défaut et celui fine-tuné. Cela signifie que le fine-tuning réalisé a bien eu un impact sur les poids du modèle, et les a plus orientés vers les thématiques, le vocabulaire et le style d'écriture des articles scientifiques.

- **GPT-2 générique**
  - Loss validation : ~3.97
  - Loss test : ~3.95
  - Accuracy validation : ~0.34 (34%)
  - Accuracy test : ~0.34 (34%)
- **GPT-2 spécialisé - modèle base avec 1 epochs par entrainement**
  - Loss validation : ~2.88
  - Loss test : ~2.86
  - Accuracy validation : ~0.44 (44%)
  - Accuracy test : ~0.45 (45%)
- **GPT-2 spécialisé - modèle base avec 3 epochs par entrainement**
  - Loss validation : ~2.78
  - Loss test : ~2.76
  - Accuracy validation : ~0.46 (46%)
  - Accuracy test : ~0.46 (46%)

On constate un grand écart de précision (~10 % de différence) et de perte/échec (env. 1,10 de différence) entre le modèle générique et les modèles spécialisés. Entre les modèles spécialisés, la différence est moins importante mais reste progressive en qualité.

Elle souligne donc l'impact positive du fine-tuning sur le modèle GPT-2, orientant les poids du modèle vers des réponses similaires aux divers articles scientifiques du jeu de données.

---
## Objectifs pour optimiser le modèle

Par la suite, nous envisageons d'optimiser le modèle en essayant d'utiliser des modèles GPT-2 plus performants, avec plus de couches en réseaux de neurones, ainsi que de paramètres. Comme il a été employé dans ce rapport, nous expérimentons et menons plusieurs entrainement différents sur différents niveaux de modèles (gpt2 base, medium, large), et des paramètres différents (différence epochs 1 et 3).

A l'avenir, un des objectifs serait de mener des recherches pour améliorer la génération de texte, comme par exemple en évitant les cas de répétition, ou encore la rapidité d'exécution du modèle.

Nous étudions certaines pistes, notamment avec le "Parameter-efficient fine-tuning" du modèle GPT-2 avec LoRA (Low-Rank Adaptation), qui pourraient réduire le temps d'entrainement et la consommation GPU en conservant la qualité des réponses, ce qui pourrait nous permettre d'envisager des modèles plus importants comme GPT-2 large.
