# files
 Source code and instances used in the paper "Learning to Run Primal Heuristics and Cutting Planes for Mixed Integer Programming".

 All instances used in the experiments are in the "instances" folder.

 All inputs needed to run the recommendation system are in the "input-recsys" folder.
 
 To run the source code of the recommendation system, simply execute the following command:
   python3 main-recsys.py

 All inputs needed to run the Boruta algorithm are in the “input-boruta” folder.

 To run the source code of the Boruta algorithm, it is necessary to inform which configuration (1, 2, 3 or 4) you want to find the most relevant features. For example:
   python3 main-boruta.py config1

 To run the diving heuristics source code, the command to be executed must contain the following parameters (in this order):
- the name of the instance;
- the id of the diving heuristic;
- 1 to activate the feasibility pump heuristic (0 otherwise).
- 1 to activate cutting planes (0 otherwise);

For example:
  python3 main-diving.py air03.mps.gz 5 1 0

Consider the following heuristic ids:
 id 0: fractional
 id 1: coefficient
 id 2: vectorLenght
 id 3: lineSearch
 id 4: conflicts
 id 5: modifiedDegree
 id 6: down
 id 7: up
