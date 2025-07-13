TDMR_EXTRACTION_PROMPT = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 

{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_07_02 = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 
2. If you will find multiple results for given triplet, please extract all results in the form of a list, so in the 'Result' dict will be a list containing all values. 
3. **DO NOT CREATE** a dictionary with keys such as "models" and the corresponding results as a value for Results section, you **SHOULD** put a value in "Result" section 
(or list of values if multiple were found) 
4. If the results values have signs such as "±" or similar, please ignore these chars and extract only the main part (value). For example if result value is defined like "90.23 ± 0.16", please extract only "90.23" out of it. 
{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_system_prompt = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 
2. If you will find multiple results for given triplet, please extract all results in the form of a list, so in the 'Result' dict will be a list containing all values. 
3. **DO NOT CREATE** a dictionary with keys such as "models" and the corresponding results as a value for Results section, you **SHOULD** put a value in "Result" section 
(or list of values if multiple were found) 
4. If the results values have signs such as "±" or similar, please ignore these chars and extract only the main part (value). For example if result value is defined like "90.23 ± 0.16", please extract only "90.23" out of it. 
"""

TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_prompt = """
Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}
"""

TDMR_EXTRACTION_PROMPT_05_04 = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 
2. If you will find multiple results for given triplet, please extract all results in the form of a list, so in the 'Result' dict will be a list containing all values. 
3. **DO NOT CREATE** a dictionary with keys such as "models" and the corresponding results as a value for Results section, you **SHOULD** put a value in "Result" section 
(or list of values if multiple were found) 
4. You **MUST** py attention to the provided caption of table and the content of a table itself. If a table does not 
contain a relevant information for provided triplet, then please output an empty dict. 
{format_instructions}
"""


TABLE_DECISION_PROMPT = """
You are provided with a triplet consisting of:
- A **Dataset**
- A **Task**
- A **Metric**

You are also given:
- A **table** containing results
- A **caption** describing the table

Your task is to decide **whether or not** the provided table contains information that is relevant to the given triplet. 

Please consider both the **content of the table** and the **caption** in making your decision. Relevant information means that the table contains numerical results or measurements that correspond to the specified dataset, task, and metric.

### Inputs:

Triplet:
{triplet}

Table:
{table}

Caption:
{table_caption}

### Guidelines:
1. If the table clearly includes a result that corresponds to the combination of dataset, task, and metric, respond with a decision of **"True"**.
2. If the table does **not** include relevant results (e.g., different dataset, irrelevant task, metric not mentioned), respond with a decision of **"False"**.
3. If you are unsure or cannot determine the relevance from the given information, respond with a decision of **"True"**.
4. Always provide a **brief explanation** (2–4 sentences) for your decision based on the input.

{format_instructions}
"""

extract_table_id_prompt = """
You will be provided with table caption. Your task is to extract table name which is included within the provided caption. 
Please extract the whole name for example from " "Table 1: Dataset statistics." the table name is "Table 1"
Please output only output table name, without any further explanation
Here is table_caption: 
{table_caption}
Your answer:
"""

TABLE_DECISION_PROMPT_WITH_PARAGRAPHS = """
You are provided with a triplet consisting of:
- A **Dataset**
- A **Task**
- A **Metric**

You are also given:
- A **table** containing results
- A **caption** describing the table
- Three paragraphs of surrounding text:
  - The paragraph mentioning the table
  - The paragraph immediately preceding it
  - The paragraph immediately following it

Your task is to decide **whether or not** the provided table contains information that is relevant to the given triplet. 

Please use all provided context — **the table content, the caption, and the surrounding paragraphs** — to make your decision.
Relevant information means that the table contains numerical results or measurements that correspond to the specified dataset, task, and metric.

### Inputs:

Triplet:
{triplet}

Table:
{table}

Caption:
{table_caption}

Paragraph before the table:
{paragraph_before}

Paragraph mentioning the table:
{paragraph_with_table}

Paragraph after the table:
{paragraph_after}

### Guidelines:
1. Analyze the table, the caption, and the surrounding paragraphs carefully.
2. If the table clearly includes a result that corresponds to the dataset, task, and metric specified in the triplet, respond with a decision of **"True"**.
3. If the table does **not** include relevant results (e.g., different dataset, irrelevant task, metric not mentioned), respond with a decision of **"False"**.
4. If you are unsure or cannot determine the relevance based on the given context, respond with a decision of **"True"**.
5. Always provide a **brief explanation** (2–5 sentences) justifying your decision based on the provided inputs.

### Output Format:
{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_WITH_PARAGRAPHS = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric, and Task) and a table with the results alongside its caption. 
Additionally, you are provided with three paragraphs:
- One paragraph that mentions the table
- One paragraph immediately preceding it
- One paragraph immediately following it

Your task is to assign a value to the extracted dataset, metric, and task triplet based on the provided data. 
Please output an updated dictionary with this result (the final dictionary should consist of: dataset, metric, task, and extracted result).

Please note:
- If the table caption does not explicitly state the dataset for the given triplet, but the surrounding paragraphs indicate a match, you should assume the dataset matches and proceed with the extraction.
- Pay close attention to both the table, its caption, and the surrounding paragraphs to ensure that you extract the correct result.

Here is the extracted triplet:
{triplet}

Here is the table with results:
{table}

Here is the table caption:
{table_caption}

Here are the paragraphs:
- Paragraph before the table:
{paragraph_before}

- Paragraph mentioning the table:
{paragraph_with_table}

- Paragraph after the table:
{paragraph_after}

Here are some guidelines: 
1. If you cannot find information about the result for the specific triplet, output an empty dictionary `{{}}`.
2. If you find multiple results matching the triplet, extract **all results** in the form of a **list**. In this case, the "Result" field should contain the list of values.
3. **DO NOT create** a dictionary where models are keys and their results are values. Only put the extracted value(s) into the "Result" field (or a list if there are multiple).
4. If a value needs to be matched approximately (e.g., slight variation in naming of metric or task), use your best judgment based on context.

{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_05_07_system_prompt_with_selecting_value = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here are some guidelines: 
1. Extract the result for only the best results obtained by proposed methods of the paper not baselines.
2. If you cannot find the information about the result for specific triplet, output empty dict. 
3. If you will find multiple results for given triplet, please extract SOTA (state of the art) result (the best one). 
4. You **SHOULD** also process the result if needed, so it's valid numeric value, for example if the results values have signs such as "±", "/" or similar 
please extract the main value. For example in case of "40±12" please extract 40. 
"""