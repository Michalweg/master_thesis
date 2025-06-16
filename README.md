# Autonomous Leaderboard generation 

## Setting up venv
1. Create a virtual env 
2. Run pip -r requirements.txt command 

## Papers parsing
To parse papers please follow next steps: 
1. Create a directory called 'parsing_experiments'
2. Inside 'parsing_experiments' directory, please create one more directory to which the outputs will be saved 
3. Go to main.py file and update the path in line 19 so it matches the desired directory that you created in the previous step
4. Run the main.py script 

The overall current flow of main.py file: 
1. A paper (in form of PDF file is being parsed by Marker package to generate markdown)
2. Papers section are extracted into the dictionary with section name as key and corresponding text as value
3. Tables from PDF are being extracted using Tabula package
4. Tables are being extracted using manual approach based on Regex
5. Finally, tables are being extracted and parsed by open source LLMs

## Current flow: 
1. Creation of triplets -> Please run the file called openai_client -> It will create a for each paper a directory with severa JSON files that contain extracted triplets
2. Normalization of triplets -> Please run the file called triplets_normalization.py it will create a normalized triplets using full determined setting (using ground truth data)
3. Extracting results for extracted triplets -> Please run the file called tdmr_extraction_without_author_approach.py which will try to assign for each triplet and for each extracted table in the paper a result value (please note that for his step you need extracted tables which are extracted independently (I'll share the folder in the email))