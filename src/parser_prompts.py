PARSER_PROMPT_TEMPLATE = """
You are a model designed to extract tables from markdown files and output them in a structured JSON format. 
Each table should be converted into CSV format, and the JSON keys should follow the pattern `table_1`, `table_2`, etc. 
Here is how the process should work:

1. Identify and extract all tables in the markdown file.
2. Convert each table to CSV format.
3. Store each CSV table in a JSON object where the key is `table_1`, `table_2`, and so on.
4. Return the JSON object as output.

Use the following format for the JSON output:
```json
{
  "table_1": "<csv_table_1>",
  "table_2": "<csv_table_2>",
  ...
}

Here are some examples you should follow:

### EXAMPLE 1
Input: 
# Sample Markdown

This is some text above the table.

| Name  | Age | City    |
|-------|-----|---------|
| Alice | 30  | New York|
| Bob   | 25  | Chicago |

Some more text after the table.


# Expected Output:
{
  "table_1": "Name,Age,City\nAlice,30,New York\nBob,25,Chicago"
}

### EXAMPLE 2

# Input:
Here’s a table of products:

| Product    | Price | Quantity |
|------------|-------|----------|
| Laptop     | 1000  | 5        |
| Smartphone | 600   | 10       |

Here’s another table of customers:

| Customer | Country   |
|----------|-----------|
| Jane     | USA       |
| Carlos   | Spain     |

# expected Output:
{
  "table_1": "Product,Price,Quantity\nLaptop,1000,5\nSmartphone,600,10",
  "table_2": "Customer,Country\nJane,USA\nCarlos,Spain"
}

Your task:
Extract the tables from the following markdown and return them in the JSON format as described.
DO NOT generate any sort of Python script

Markdown content from which you should extract tables:
You MUST use the markdown below to extract tables
<markdown_file_content>
"""

# Define in more details the content of .csv file (create constraints to limit the possibility of handling sth unexpected)
phi3_medium_prompt_template = """
You will be given content of a markdown file for which you should extract all tables. 
Please extract the tables as JSON file in which keys are in form of "table_1", "table_2" and corresponding values are strings in form of .csv file containing content of extracted tables.
DO NOT describe the tables, just extract it from the given markdown file!

Here is the content of a markdown file: 
{markdown_file_content}

Make sure that you structure the response in desired format
"""

phi3_medium_prompt_template_specific_csv = """
You will be given content of a markdown file for which you should extract all tables. 
If you cannot find any tables in the provided markdown files content, please output empty string. 
Otherwise, please extract the tables as JSON file in which keys are table_names and corresponding values are strings in form of .csv file containing content of extracted tables.
Please find an example of a correct output structure below:
<correct_structured_output>
{{
  "table_1": "A,B,C\n1,2,3\n4,5,6"
}}
</correct_structured_output> 

Follow below instructions: 
<instructions>
1. DO NOT describe the tables, just extract it from the given markdown file!
2. DO NOT return table in "column_name": "list_of_vales" format!
3. Make sure that you structure the response in desired format
4. Please include ONLY the extracted tables in the output!
</instructions>

Here is the content of a markdown file: 
{markdown_file_content}
"""

gpt_4_prompt = """
Human: You will be given content of a markdown file for which you should extract all tables along with tables' captions. 
If you cannot find any tables in the provided markdown files content, please output empty string.
Otherwise, please extract the tables as JSON file in which keys are table_names and corresponding values are strings in form of .csv file containing content of extracted tables.
Please find an example of a correct output structure below:
<correct_structured_output>
{{
  "caption": "table caption", 
  "data": "A,B,C\n1,2,3\n4,5,6"
}}
</correct_structured_output> 

Follow below instructions: 
<instructions>
1. DO NOT describe the tables, just extract it from the given markdown file!
2. DO NOT return table in "column_name": "list_of_vales" format!
3. Make sure that you structure the response in desired format
4. Please include ONLY the extracted tables in the output!
5. If you cannot find any tables in the provided markdown files content, please output empty string without any explanation.
6. You MUST extract all tables!
7. When a cell spans multiple rows, its value will be copied to each of those rows. Similarly, if a cell spans multiple columns, its value will be copied to each of those columns.
</instructions>

Here is the content of a markdown file: 
</markdown_file_content>
{markdown_file_content}
</markdown_file_content>

Assistant:
"""

gpt_4o_prompt_rana = """
Act as data analyst and extract all the tables data along with the tables' captions from the uploaded document in the JSON format.
Adhere to the following instructions while generating the response:
- The output will be a JSON object that is a list of dictionaries
- Each dictionary represents a table that has two keys: "caption" for storing table's caption and "data" for storing data related to the table
- Table captions will include the table identifier in the document ((e.g., "Table 1", "Table 2" etc)
- Any associated text such as table descriptions should be included as part of the table caption
- Both column and row headers will be extracted
- Properly handle tables that do not explicitly have row headers, but have row labels within the first column.
- if tables have a hierarchy of headers, then preserve the hierarchy
- Table cell data will be extracted as raw text
- When a cell spans multiple rows, its value will be copied to each of those rows. Similarly, if a cell spans multiple columns, its value will be copied to each of those columns.
- Make sure that you apply rule above, limits the no. of empty values to minimum. 
- All tables will be treated with equal priority
- No tables should be skipped
- All data for the response will come from the provided document, and not from external sources
- Other surrounding content, like paragraphs before or after the tables will not be extracted
- Response output should just limited to the JSON object, nothing else.
- If you cannot extract any of the tables, please output an empty string.
- You MUST extract all tables!

Uploaded document:
<uploaded_document>
{uploaded_document_id}
</uploaded_document>
"""

gpt_4o_prompt_rana_without_document = """
Act as data analyst and extract all the tables data along with the tables' captions from the uploaded document in the JSON format.
Adhere to the following instructions while generating the response:
- The output will be a JSON object that is a list of dictionaries
- Each dictionary represents a table that has two keys: "caption" for storing table's caption and "data" for storing data related to the table
- Table captions will include the table identifier in the document ((e.g., "Table 1", "Table 2" etc)
- Any associated text such as table descriptions should be included as part of the table caption
- Both column and row headers will be extracted
- Properly handle tables that do not explicitly have row headers, but have row labels within the first column.
- The extracted row should have a structure of "column_name": "row_value" for every column detected. 
- Please make sure that you included all columns from the table. Sometimes tables have column header, please include them in the column name for example: {"POS, Dev, Acc.": "96.56"} where POS is a column header, "Dev" and "Acc." are column names.
- If table contains also row header, then please create an empty column for that, for example: {"": "Yes", "POS, Train": "98.46"}
- if tables have a hierarchy of headers, then preserve the hierarchy
- Table cell data will be extracted as raw text
- When a cell spans multiple rows, its value will be copied to each of those rows. Similarly, if a cell spans multiple columns, its value will be copied to each of those columns.
- Make sure that you apply rule above, limits the no. of empty values to minimum. 
- All tables will be treated with equal priority
- No tables should be skipped
- All data for the response will come from the provided document, and not from external sources
- Other surrounding content, like paragraphs before or after the tables will not be extracted
- Response output should just limited to the JSON object, nothing else.
- If you cannot extract any of the tables, please output an empty string.
"""

gpt_4o_prompt_rana_without_document_output_csv = """
Act as data analyst and extract all the tables data along with the tables' captions from the uploaded document in the JSON format.
Adhere to the following instructions while generating the response:
- The output will be a JSON object that is a list of dictionaries
- Each dictionary represents a table that has two keys: "caption" for storing table's caption and "data" for storing data related to the table
- Inside the "data" key please include extracted table in .csv format 
- Table captions will include the table identifier in the document ((e.g., "Table 1", "Table 2" etc)
- Any associated text such as table descriptions should be included as part of the table caption
- Both column and row headers will be extracted
- Properly handle tables that do not explicitly have row headers, but have row labels within the first column.
- if tables have a hierarchy of headers, then preserve the hierarchy
- Table cell data will be extracted as raw text
- When a cell spans multiple rows, its value will be copied to each of those rows. Similarly, if a cell spans multiple columns, its value will be copied to each of those columns.
- Make sure that you apply rule above, limits the no. of empty values to minimum. 
- All tables will be treated with equal priority
- No tables should be skipped
- All data for the response will come from the provided document, and not from external sources
- Other surrounding content, like paragraphs before or after the tables will not be extracted
- Response output should just limited to the JSON object, nothing else.
- If you cannot extract any of the tables, please output an empty string.
"""


llama_31_7B_prompt = """
Human: You will be given content of a markdown file for which you should extract all tables. 
Otherwise, please extract the tables as JSON file in which keys are table_names and corresponding values are strings in form of .csv file containing content of extracted tables.
Please find an example of a correct output structure below:
<correct_structured_output>
{{
  "table_1": "A,B,C\n1,2,3\n4,5,6"
}}
</correct_structured_output> 

Follow below instructions: 
<instructions>
1. DO NOT describe the tables, just extract it from the given markdown file!
2. DO NOT return table in "column_name": "list_of_vales" format!
3. Make sure that you structure the response in desired format
4. Please include ONLY the extracted tables in the output!
5. If you cannot find any tables in the provided markdown files content, please output empty string without any explanation.
6. You MUST extract all tables!
7. Do not create Your own tables if there are none in the provided file!
</instructions>

Here is the content of a markdown file: 
</markdown_file_content>
{markdown_file_content}
</markdown_file_content>

Assistant:
"""


triplets_extraction_prompt_llama_3_1 = """
Human: You will be given a part of a research paper as input. 
Please extract different tuples including the name of the task addressed in the paper, utilized datasets and evaluation metrics.
Please use json format for each different tuple. Example format: [{{"Task": "Task name", "Dataset": "Dataset name", "Metric": "Metric name"}}]. 
Your answer will immediately start with the json object satisfying the given template and contain nothing else.
If you cannot find any of such tuples, return empty string without any explanation.
If you cannot find all of required fields, skip them, extracted tuples must contain all of the fields.

Part of a research paper: 
<part_of_research_paper>
{part_of_research_paper}
</part_of_research_paper>


Assistant:
"""


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

system_prompt_for_triplet_extraction_gpt4o = \
"""
You are a helpful assistant. 
"""