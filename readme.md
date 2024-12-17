## SciDQA: A Deep Reading Comprehension Dataset over Scientific Papers 

### Abstract 
Scientific literature is typically dense, requiring significant background knowledge and deep comprehension for effective engagement. We introduce SciDQA, a new dataset for reading comprehension that challenges LLMs for a deep understanding of scientific articles, consisting of 2,937 QA pairs. Unlike other scientific QA datasets, SciDQA sources questions from peer reviews by domain experts and answers by paper authors, ensuring a thorough examination of the literature. We enhance the dataset's quality through a process that carefully filters out lower quality questions, decontextualizes the content, tracks the source document across different versions, and incorporates a bibliography for multi-document question-answering. Questions in SciDQA necessitate reasoning across figures, tables, equations, appendices, and supplementary materials, and require multi-document reasoning. We evaluate several open-source and proprietary LLMs across various configurations to explore their capabilities in generating relevant and factual responses. Our comprehensive evaluation, based on metrics for surface-level similarity and LLM judgements, highlights notable performance discrepancies. SciDQA represents a rigorously curated, naturally derived scientific QA dataset, designed to facilitate research on complex scientific text understanding.  
[Paper URL](https://arxiv.org/abs/2411.05338)  

### Setting up the repo:  
`conda create -n scidqa --python=3.11`  
`conda activate scidqa`  
`pip install -r requirements.txt` 

### How to use the dataset:
To use the QA dataset, load it as dataframe using pandas:  
```
import pandas as pd
scidqa_df = pd.read_xlsx('src/data/scidqa.xlsx')
print(scidqa_df.columns)
```

The paper metadata (title and abstract) is present in `src/data/relevant_ptabs.pkl` and can be used as follows:  
```
import pickle
paper_id = scidqa_df['pid'][0]
with open('src/data/relevant_ptabs.pkl', 'rb') as fp:
    papers_tabs = pickle.load(fp)

print('Paper title: ', papers_tabs[paper_id]['title'])
print('Paper abstract: ', papers_tabs[paper_id]['abs'])
```

To use the full-text of papers for the QA pairs, use the `src/data/papers_fulltext_nougat.pkl` file. It can be used as follows:  
```
import pickle
paper_id = scidqa_df['pid'][0]
with open('src/data/papers_fulltext_nougat.pkl', 'rb) as fp:
    paper_fulltext_dict = pickle.load(fp)

print("Full-text of the mansucript at submission:\n", paper_fulltext_dict[paper_id]['initial'])
print("Full-text of the camera-ready mansucript:\n", paper_fulltext_dict[paper_id]['final'])
```


SciDQA data is also available on [HF](https://huggingface.co/datasets/yale-nlp/SciDQA) and can be used as follows:  
```
from datasets import load_dataset
scidqa = load_dataset("yale-nlp/SciDQA")
```

### Dataset License
Our dataset is made available under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/).

### Citation
If you use our dataset, please cite our paper:  
```
@inproceedings{singh-etal-2024-scidqa,
    title = "{S}ci{DQA}: A Deep Reading Comprehension Dataset over Scientific Papers",
    author = "Singh, Shruti  and
      Sarkar, Nandan  and
      Cohan, Arman",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    url = "https://aclanthology.org/2024.emnlp-main.1163",
    abstract = "Scientific literature is typically dense, requiring significant background knowledge and deep comprehension for effective engagement. We introduce SciDQA, a new dataset for reading comprehension that challenges language models to deeply understand scientific articles, consisting of 2,937 QA pairs. Unlike other scientific QA datasets, SciDQA sources questions from peer reviews by domain experts and answers by paper authors, ensuring a thorough examination of the literature. We enhance the dataset{'}s quality through a process that carefully decontextualizes the content, tracks the source document across different versions, and incorporates a bibliography for multi-document question-answering. Questions in SciDQA necessitate reasoning across figures, tables, equations, appendices, and supplementary materials, and require multi-document reasoning. We evaluate several open-source and proprietary LLMs across various configurations to explore their capabilities in generating relevant and factual responses, as opposed to simple review memorization. Our comprehensive evaluation, based on metrics for surface-level and semantic similarity, highlights notable performance discrepancies. SciDQA represents a rigorously curated, naturally derived scientific QA dataset, designed to facilitate research on complex reasoning within the domain of question answering for scientific texts.",
}
```