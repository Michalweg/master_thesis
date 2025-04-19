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
{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_05_04 = \
"""
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