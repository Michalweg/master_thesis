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


triplets_extraction_prompt_gpt_4_turbo_more_context = """
You are an expert data analyst specializing in machine learning literature.
You will be given a parts of research papers from the Machine Learning domain as an input
Please extract different triplets including the name of the task addressed in the paper, utilized datasets and evaluation metrics. 
Please focus on the items for which then results were obtained (for given task, on the defined dataset using metric)



Follow all instructions below precisely.

<instructions>

General Rules:

1. Extract every unique combination of (Task, Dataset, Metric) found. If a single experiment uses multiple metrics (e.g., F1 and Recall), create a separate JSON object for each metric.
2. Only output triplets where all three fields (Task, Dataset, Metric) are explicitly found and meet the criteria below. If any field is missing or invalid for a potential triplet, discard the entire triplet.
3. If no valid triplets that satisfy all rules can be found, return an empty list []. Additionally, if not complete triplet was found, then **DO NOT** report, output only complete triplets.  

</instructions>
"""

triplets_extraction_prompt_gpt_4o = """
Here is the part of the research paper:
{research_paper}
"""