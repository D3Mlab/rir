

# Self-supervised Contrastive BERT Fine-tuning for Fusion-Based Reviewed-Item Retrieval

[link](https://link.springer.com/chapter/10.1007/978-3-031-28244-7_1)

- [Setup](#Setup)
- [Training](#Training)
- [Inference](#Inference)
  - Pretrained langauge models for RIR
  - BM25
  - TF IDF
- [Citation](#Citation)

## Abstract
As natural language interfaces enable users to express increasingly complex natural language queries, there is a parallel explosion of user review content that can allow users to better find items such as restaurants, books, or movies that match these expressive queries. While Neural Information Retrieval (IR) methods have provided state-of-the-art results for matching queries to documents, they have not been extended to the task of Reviewed-Item Retrieval (RIR), where query-review scores must be aggregated (or fused) into item-level scores for ranking. In the absence of labeled RIR datasets, we extend Neural IR methodology to RIR by leveraging self-supervised methods for contrastive learning of BERT embeddings for both queries and reviews. Specifically, contrastive learning requires a choice of positive and negative samples, where the unique two-level structure of our item-review data combined with meta-data affords us a rich structure for the selection of these samples. For contrastive learning in a Late Fusion scenario (where we aggregate query-review scores into item-level scores), we investigate the use of positive review samples from the same item and/or with the same rating, selection of hard positive samples by choosing the least similar reviews from the same anchor item, and selection of hard negative samples by choosing the most similar reviews from different items. We also explore anchor sub-sampling and augmenting with meta-data. For a more end-to-end Early Fusion approach, we introduce contrastive item embedding learning to fuse reviews into single item embeddings. Experimental results show that Late Fusion contrastive learning for Neural RIR outperforms all other contrastive IR configurations, Neural IR, and sparse retrieval baselines, thus demonstrating the power of exploiting the two-level structure in Neural RIR approaches as well as the importance of preserving the nuance of individual review content via Late Fusion methods.

## [Setup](#Setup)
```commandline
git clone https://github.com/D3Mlab/rir
pip install -r requirements.txt
```
### On Colab

```
!git clone https://github.com/D3Mlab/rir
!pip install transformers
```
```
import os
os.environ['PYTHONPATH'] += ":/content/rir"
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
Create a folder name 'data' and copy PMD.csv and the files in the 'review' folder from [RIRD dataset](https://github.com/D3Mlab/rir_data) in the 'data' directory

You can also produce embedding .pkl file which is needed for least similar positive and most similar negatives sampling methods, it will create 'BERT_embedded_reviews.pkl', put it under the folder data after it has been created (you can get it from [RIRD dataset](https://github.com/D3Mlab/rir_data) as well)
```commandline
python ./Neural_PM/runnersembedder_cmd.py --tpu --train_data_path './data/50_above3.csv'
```

## [Training](#Training)  

### Base CLFR
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --positive_pair_per_restaurant 100 --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --tpu when training would be on TPU, and --gpu for GPU otherwise it would be on CPU (trainng on CPU is not recommended since it take so long)
- --epochs spefies the maximum number of epochs 
- --run_neuralpm indicates to run RIR inference after the self-supervised learning is done
- --patience is the patience parameter for the early stopping
- --repeat specifies number of times the experiment should be repeated 
- --change_seed means to change the random seed in each experiment repeat
- --true_labels_path is path of the label data
- --positive_pair_per_restaurant is number of positive review pairs from each item, we use specific number of them so training would be balanced for all items

### Least Similar Positives
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --least_similar --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --least_similar select least similar pairs as positives pairs for each item (restaurant)
### Same Rating Positives
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --same_rating  --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --same_rating select same rating pairs as positives pairs for each item (restaurant)
### Same Rating and Least Similar Positives
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --least_similar --same_rating --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```

### Hard Negatives
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --hard_negative --hard_negative_num 1 --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --hard_negative specifies to add hard_negatives in addition to in-batch negatives
- --hard_negative_num specifies number of hard negatives

### IR Style Training 
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --patience 8 --ir_style --ir_sampling_type 'RC' --true_labels_path './data/PMD.csv' --change_seed --repeat 5
```
- --ir_style indicated to use ir style contrastive learning (not using the two level structure)
- --ir_sampling_type: RC is the random sampling and IC is the independent cropping (refer to paper for details)


### Data Augmentation

#### Anchor Subsampling (Span)
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --subsample_query --positive_pair_per_restaurant 100 --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --subsample_query subsamples the query to a span in the training
#### Anchor Subsampling (Sentence)
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --subsample_query_sentence --positive_pair_per_restaurant 100 --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- subsample_query_sentence subsamples the query to a sentence in the training

#### Prepending Meta Data (PPMD)
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --epochs 50 --run_neuralpm --prepend_neuralpm --patience 8 --change_seed --repeat 5 --true_labels_path './data/PMD.csv'
```
- --prepend_neuralpm prepends the category type of item to review text in the inference

### Item Embeddings (Our Early Fusion)
### Save Warmups
We warm up item embedding with average of review embedding, this command creates the .pkl file result for it named 'item_embeddings_BERT.pickle'
```commandline
python ./Neural_PM/runnerssave_warmups.py --tpu 
```

### Learning Item Embedding
```commandline
python ./Neural_PM/runnerstrainer_cmd.py --tpu --item_embedding --warmup --warmup_weights './item_embeddings_BERT.pickle' --epochs 50 --positive_pair_per_restaurant 300 --run_neuralpm --patience 8
```
- --item_embedding indicates to learn item embeddings
- --warmup_weights specifies the address for war up weights file


## [Inference](#Inference)

### Pretrained langauge models for RIR
```commandline
#### If you don't have a TPU

python ./Neural_PM/runners/runner_cmd.py --from_pt --true_labels_path ./data/PMD.csv --model_name Luyu/condenser --tokenizer_name Luyu/condenser

#### If you have a TPU

python ./Neural_PM/runners/runner_cmd.py --tpu --from_pt --true_labels_path ./data/PMD.csv --model_name Luyu/condenser --tokenizer_name Luyu/condenser

```
- --tokenizer_name and --model_name specify the model name and tokenizer name, it should be from https://huggingface.co/models or a local path
- --from_pt when the original model is on Pytorch on Huggingface (that's usually the case, so always pass this argument)
- "Luyu/condenser" for condenser 
- "Luyu/co-condenser-wiki" for CoCondenser
- "bert-base-uncased" for BERT

### BM25
```commandline
python ./Neural_PM/runnersrunner_BM25.py  --true_labels_path './data/PMD.csv' 
```

### TF IDF
```commandline
python ./Neural_PM/runnersrunner_cmd.py --tf_idf --true_labels_path './data/PMD.csv' 
```




## [Citation](#Citation)
Cite this work using the Bibtex below:
```
@inproceedings{abdollah2023self,
  title={Self-supervised Contrastive BERT Fine-tuning for Fusion-Based Reviewed-Item Retrieval},
  author={Abdollah Pour, Mohammad Mahdi and Farinneya, Parsa and Toroghi, Armin and Korikov, Anton and Pesaranghader, Ali and Sajed, Touqir and Bharadwaj, Manasa and Mavrin, Borislav and Sanner, Scott},
  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part I},
  pages={3--17},
  year={2023},
  organization={Springer}
}
```

