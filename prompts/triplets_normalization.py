normalization_system_prompt = """
You will be given with a data item and a list of similar items. Your task is to decide, whether this provided data item is a new data item
or whether it's already defined in different name in the provided list. If it's a new item (it does not exist in the provided list), please return None. 
If the provided list already contains this data item in different name, please provide the defined version.

<instructions>
1. Do not provide any further explanations!
2. Please take a note that the new item doesn't have to be exactly 1:1 with the defined list, it's more about the meaning.
3. Additionally, please make sure that when the data item already exists you MUST provide the value from the defined list, not the original data item.
4. DO NOT PROVIDE ANY EXPLANATION!
</instructions>
"""


normalization_system_prompt_gpt_4_tubo = """
You will be given with a data item and a list of similar items. Your task is to decide, whether this provided data item is a new data item
or whether it's already defined in different name in the provided list. If it's a new item (it does not exist in the provided list), please return None. 
If the provided list already contains this data item in different name, please provide the defined version.

<instructions>
1. Do not provide any further explanations!
2. Please take a note that the new item doesn't have to be exactly 1:1 with the defined list, it's more about the meaning.
3. Additionally, please make sure that when the data item already exists you MUST provide the value from the defined list, not the original data item.
4. DO NOT PROVIDE ANY EXPLANATION!
</instructions>

Return the output in the following JSON format:
```json
{
"output_data_item": "output data item from the provided data items list",
}
'''
"""

normalization_user_prompt = """
Here is the data item to analyze (input):
{data_item}

Here is the defined data items list:
{defined_list}
"""