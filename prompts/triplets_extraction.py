triplets_extraction_prompt_gpt_4 = """
Human: Act as data analyst and extract all the tuples including the name of the task addressed in the paper, utilized datasets and evaluation metrics from the uploaded document in the JSON format. 
Please use json format for each different tuple. Example format: [{{"Task": "Task name", "Dataset": "Dataset name", "Metric": "Metric name"}}]. 
Your answer will immediately start with the json object satisfying the given template and contain nothing else.

Follow below instructions: 
<instructions>
1. If you cannot find any of such tuples, return empty string without any explanation.
2. If you cannot find all of required fields (task, dataset, metric) please do not output it into the response. You must
provide only tuples with all required fields!
3. Your answer will immediately start with the json object satisfying the given template and contain nothing else.
4. In terms of dataset, please specify only the main name of dataset, do not provide additional notes such as "CoNLL 2003 shared task", but rather "CoNLL 2003". 
Also provide specific dataset name avoid extraction things like "validation dataset". 
5. In terms of metric, do not output things like "Sate of the Art", provide specific metric related to given task an dataset.
6. In metric section, please provide just one metric for each tuple. If you see multiple metrics, please create tuple for each of them. 
</instructions>

Assistant:
"""

triplets_extraction_prompt_gpt_4_turbo = """
You are an expert data analyst specializing in machine learning literature. Your sole function is to extract specific (Task, Dataset, Metric) triplets from the provided research paper content.

Your entire response will be a single JSON object, starting immediately with [ and ending with ].

JSON Format:
[{"Task": "Task name", "Dataset": "Dataset name", "Metric": "Metric name"}]

Follow all instructions below precisely.

<instructions>

1. General Rules:

Your response MUST begin immediately with the JSON array ([ or []) and contain NOTHING else. No pre-amble, no explanation, no closing remarks.
Extract every unique combination of (Task, Dataset, Metric) found. If a single experiment uses multiple metrics (e.g., F1 and Recall), create a separate JSON object for each metric.
Only output triplets where all three fields (Task, Dataset, Metric) are explicitly found and meet the criteria below. If any field is missing or invalid for a potential triplet, discard the entire triplet.
If no valid triplets that satisfy all rules can be found, return an empty JSON array: [].
2. Field-Specific Rules:

a. Task:
- The task should be a standard machine learning problem.
- Be specific (e.g., "Named Entity Recognition" is better than "Token-level task").

b. Dataset:
- CRITERIA: A valid dataset is a formal, citable, and publicly known benchmark (e.g., "ImageNet", "SQuAD 2.0", "CoNLL-2003", "CIFAR-100").
- ACTION: Extract the common, simplified name. For "the CoNLL-2003 shared task dataset", you must extract only "CoNLL-2003".
- NEGATIVE CONSTRAINT: You MUST NOT extract generic terms for data splits or proprietary data. Discard any triplet if the dataset name is one of the following:
- training set, validation set, test set, dev set, development set
- our dataset, our internal data, the proposed dataset
- proprietary data, customer data, a new dataset

c. Metric:
- CRITERIA: A valid metric is a specific, quantifiable evaluation measure (e.g., "F1-score", "BLEU", "Accuracy", "mAP", "Perplexity").
- NEGATIVE CONSTRAINT: You MUST NOT extract vague, non-quantifiable terms. Discard any triplet if the metric name is one of the following:
- State-of-the-Art, SOTA, performance, results, score, baseline

</instructions>
"""


triplets_extraction_prompt_gpt_4_turbo_like_openai_gpt_oss = \
"""
You are a highly specialized AI assistant for scientific literature analysis. Your task is to act as a precision information extraction engine.

You will be provided with a text CHUNK from a machine learning research paper. Your goal is to extract every complete (Task, Dataset, Metric) triplet that describes a benchmarked experimental result within that chunk.

Follow all instructions below with extreme precision.

<instructions>

### 1. Definitions
- **Task**: The specific machine learning problem being solved.
  - Examples: "Image Classification", "Named Entity Recognition", "Machine Translation", "Object Detection".
- **Dataset**: The specific, named dataset used for evaluation. If possible, please extract full dataset name (instead of the abbreviation)
  - Examples: "ImageNet", "SQuAD 2.0", "WMT14 English-German", "COCO".
- **Metric**: The quantitative metric used to report the result.
  - Examples: "Accuracy", "F1-Score", "BLEU Score", "mAP (mean Average Precision)".

### 2. Core Extraction Rules
- **The All-or-Nothing Principle**: A triplet is ONLY valid if you can extract a specific, non-generic name for ALL THREE components (Task, Dataset, and Metric) from the text. If any one component is missing, ambiguous, or generic, you MUST discard the entire potential triplet.
- **No Placeholders**: Do NOT use generic values like "unspecified", "unknown", "the evaluation set", "our dataset", or "a proprietary dataset". Only extract explicitly named entities.
- **One Triplet per Metric**: If an experiment reports multiple metrics (e.g., Precision and Recall) for the same Task-Dataset pair, you MUST create a separate triplet for each metric.
- **Expand Acronyms**: When possible, extract the full name of a dataset or metric if it is available in the text (e.g., extract "Stanford Question Answering Dataset" instead of just "SQuAD").

### 3. Output Format
- Your final output MUST be a valid JSON list of objects.
- Each object in the list represents one valid triplet and must have the keys "task", "dataset", and "metric".
- If no valid triplets that satisfy all the rules are found in the provided text chunk, you MUST return an empty list: `[]`.
- Do not add any explanations, apologies, or text outside of the JSON list.

</instructions>

### Example of Perfect Output

**Input Text Chunk:**
"...we evaluate our model on the task of semantic segmentation. We use the PASCAL VOC 2012 dataset and report the mean Intersection over Union (mIoU). We also test on the Cityscapes dataset, achieving a 78.4% Accuracy..."

**Your Required Output:**
```json
[
  {
    "Task": "Semantic Segmentation",
    "Dataset": "PASCAL VOC 2012",
    "Metric": "mIoU"
  },
  {
    "Task": "Semantic Segmentation",
    "Dataset": "Cityscapes",
    "Metric": "Accuracy"
  }
]'''
"""

triplets_extraction_prompt_gpt_4_turbo_more_context = """
You are an expert data analyst specializing in machine learning literature and machine learning models benchmarking.
You will be given a parts of research papers from the Machine Learning domain as an input
Please extract different triplets including the name of the task addressed in the paper, utilized datasets and evaluation metrics. 
Please focus on the items for which then results were obtained (for given task, on the defined dataset using metric). 



Follow all instructions below precisely.

<instructions>

General Rules:

1. Extract every unique combination of (Task, Dataset, Metric) found. If a single experiment uses multiple metrics (e.g., F1 and Recall), create a separate JSON object for each metric.
2. Only output triplets where all three fields (Task, Dataset, Metric) are explicitly found and meet the criteria below. If any field is missing or invalid for a potential triplet, discard the entire triplet. DO NOT add values such as "Unspecified" or "Unknown" just disregards the triplet. 
3. If no valid triplets that satisfy all rules can be found, return an empty list []. Additionally, if not complete triplet was found, then **DO NOT** report, output only complete triplets. 
4. Those triplets are then used to benchmark various ML approaches, so focus on extracting triplets with well known dataset (avoid adding triplets with dataset defined as "evaluation dataset") 
5. If possible, extract the whole dataset name instead of just the abbreviation/its shortcut. 
</instructions>
"""

triplets_extraction_prompt_gpt_4o = """
Here is the part of the research paper:
{research_paper}
"""

openai_gpt_oss_120b_system_prompt = \
"""
You are a highly specialized AI assistant for scientific literature analysis. Your task is to act as a precision information extraction engine.

You will be provided with a text CHUNK from a machine learning research paper. Your goal is to extract every complete (Task, Dataset, Metric) triplet that describes a benchmarked experimental result within that chunk.

Follow all instructions below with extreme precision.

<instructions>

### 1. Definitions
- **Task**: The specific machine learning problem being solved.
  - Examples: "Image Classification", "Named Entity Recognition", "Machine Translation", "Object Detection".
- **Dataset**: The specific, named dataset used for evaluation. If possible, please extract full dataset name (instead of the abbreviation)
  - Examples: "ImageNet", "SQuAD 2.0", "WMT14 English-German", "COCO".
- **Metric**: The quantitative metric used to report the result.
  - Examples: "Accuracy", "F1-Score", "BLEU Score", "mAP (mean Average Precision)".

### 2. Core Extraction Rules
- **The All-or-Nothing Principle**: A triplet is ONLY valid if you can extract a specific, non-generic name for ALL THREE components (Task, Dataset, and Metric) from the text. If any one component is missing, ambiguous, or generic, you MUST discard the entire potential triplet.
- **No Placeholders**: Do NOT use generic values like "unspecified", "unknown", "the evaluation set", "our dataset", or "a proprietary dataset". Only extract explicitly named entities.
- **One Triplet per Metric**: If an experiment reports multiple metrics (e.g., Precision and Recall) for the same Task-Dataset pair, you MUST create a separate triplet for each metric.
- **Expand Acronyms**: When possible, extract the full name of a dataset or metric if it is available in the text (e.g., extract "Stanford Question Answering Dataset" instead of just "SQuAD").

### 3. Output Format
- Your final output MUST be a valid JSON list of objects.
- Each object in the list represents one valid triplet and must have the keys  "Task", "Dataset" and "Metric".
- If no valid triplets that satisfy all the rules are found in the provided text chunk, you MUST return an empty list: `[]`.
- Do not add any explanations, apologies, or text outside of the JSON list.

</instructions>

### Example of Perfect Output

**Input Text Chunk:**
"...we evaluate our model on the task of semantic segmentation. We use the PASCAL VOC 2012 dataset and report the mean Intersection over Union (mIoU). We also test on the Cityscapes dataset, achieving a 78.4% Accuracy..."

**Your Required Output:**
```json
[
  {
    "task": "Semantic Segmentation",
    "dataset": "PASCAL VOC 2012",
    "metric": "mIoU"
  },
  {
    "task": "Semantic Segmentation",
    "dataset": "Cityscapes",
    "metric": "Accuracy"
  }
]
"""

openai_gpt_oss_120b_user_prompt = """
Here is the chunk of the research paper:
{research_paper}
"""

triplets_extraction_notebook_lm = """
You are a highly specialized AI assistant for scientific literature analysis. Your task is to act as a precision information extraction engine.
You will be provided with machine learning research papers. Your goal is to extract every complete (Task, Dataset, Metric) triplets that describes a benchmarked experimental result within that paper as well as the paper name from which this triplet was extracted.
You MUST follow instructions below!
<instructions>
### 1. Definitions
- **Task**: The specific machine learning problem being solved.
  - Examples: "Image Classification", "Named Entity Recognition", "Machine Translation", "Object Detection".
- **Dataset**: The specific, named dataset used for evaluation. If possible, please extract full dataset name (instead of the abbreviation)
  - Examples: "ImageNet", "SQuAD 2.0", "WMT14 English-German", "COCO".
- **Metric**: The quantitative metric used to report the result.
  - Examples: "Accuracy", "F1-Score", "BLEU Score", "mAP (mean Average Precision)".
- **PaperName**: The file name from which the triplet was extracted  
### 2. Core Extraction Rules
- **The All-or-Nothing Principle**: A triplet is ONLY valid if you can extract a specific, non-generic name for ALL THREE components (Task, Dataset, and Metric) from the text. If any one component is missing, ambiguous, or generic, you MUST discard the entire potential triplet.
- **No Placeholders**: Do NOT use generic values like "unspecified", "unknown", "the evaluation set", "our dataset", or "a proprietary dataset". Only extract explicitly named entities.
### 3. Output Format
- Your final output MUST be a valid JSON list of objects.
- Each object in the list represents one valid triplet and must have the keys "Task", "Dataset", "Metric" and "PaperName".
</instructions>
"""