![img.png](img.png)# Reviewed-Item Retrieval Dataset (RIRD)




## PMD.csv
### Examples

### Description
To address the lack of existing RIR datasets, we curated the Reviewed-Item Retrieval Dataset (RIRD)1 to support our analysis. We used reviews related to 50 popular restaurants in Toronto, Canada obtained from the Yelp dataset.2 We selected restaurants with a minimum average rating of 3 and at least 400 reviews that were not franchises (e.g., McDonalds) since we presume franchises are well- known and do not need recommendations. We created 20 queries for 5 different conceptual categories highlighted in Table1 (with examples). These 5 groups capture various types of natural language preference statements that occur in this task. We then had a group of annotators assess the binary relevance of each of these 100 queries to the 50 restaurants in our candidate set. Each review was labeled by 5 annotators and the annotations showed a kappa agreement score of 0.528, demonstrating moderate agreement according to Landis and Koch [1], which is expected given the subjectivity of the queries in this task ([2]). There is a total number of 29k reviews in this dataset.
Categories of queries and their examples.

Query category | Example
--- | --- 
Indirect queries | I am on a budget.
Queries with negation|Not Sushi but Asian.
General queries | Nice place with nice drinks.
Detailed queries | A good cafe for a date that has live music.
Contradictory queries | A fancy, but affordable place.

### Data Columns
- Restaurant name: name of restaurant (text)
- Query: query text
- Annotator1, Annotator2, Annotator3, Annotator4, Annotator5: binary relevance score of each annotator (0-> not relevant, 1-> relevant)
- If only Low or  High: aggregated binary relevance value over 5 annotators (0-> not relevant, 1-> relevant)




For more details, refer to the paper: # Self-supervised Contrastive BERT Fine-tuning for Fusion-Based Reviewed-Item Retrieval [link](https://link.springer.com/chapter/10.1007/978-3-031-28244-7_1)

## [Citation](#Citation)
Cite this dataset using the Bibtex below:
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



## References
[1] J Richard Landis and Gary G Koch. 1977. The measurement of observer agreement for categorical data. 705 biometrics, pages 159–174.

[2] Krisztian Balog, Filip Radlinski, and Alexandros Karat- zoglou. 2021. On interpretation and measurement of soft attributes for recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 890–899.


