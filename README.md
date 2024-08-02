# files
 Source code and instances used in the paper "Learning to Run Primal Heuristics and Cutting Planes for Mixed Integer Programming".

 All instances used in the experiments are in the "instances" folder.

 All inputs needed to run the recommendation system are in the "input-recsys" folder.
 
 To run the source code of the recommendation system, simply execute the following command: python3 main-recsys.py

 All inputs needed to run the Boruta algorithm are in the “input-boruta” folder.

 To run the source code of the Boruta algorithm, it is necessary to inform which configuration (1, 2, 3 or 4) you want to find the most relevant features. 
 
 For example: python3 main-boruta.py config1

 To run the diving heuristics source code, the command to be executed must contain the following parameters (in this order):
- the name of the instance;
- the ID of the diving heuristic;
- 1 to activate the feasibility pump heuristic (0 otherwise).
- 1 to activate cutting planes (0 otherwise);

For example: python3 main-diving.py air03.mps.gz 5 1 0

Consider the following heuristic IDs:
- id 0: fractional
- id 1: coefficient
- id 2: vector lenght
- id 3: linesearch
- id 4: conflicts
- id 5: modified degree
- id 6: down
- id 7: up

The "input-createds" folder contains the results obtained by each heuristic and configuration. Furthermore, it contains a file with all the instance features (and their values) provided by Python-MIP.

The source code "main-createds.py" calculates the success obtained by the heuristics in each configuration and creates the datasets with the features provided by Python-MIP. 
To run the source code and create the datasets for the four configurations, simply run the following command: python3 main-createds.py

The file "Feature description.pdf" contains the description of the 207 features provided by Python-mip package.
