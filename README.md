# Autonomous Leaderboard generation 


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