# How to reproduce the GP results:

1. To install the dependencies, please run ```pip install -r requirements.txt```
2. Run the program with ```python3.7 gp_behaviour.py```.  The results should be exaclty the same as in the report, because the same random seed is kept.
3. If you use Mac Os and have a problem with pygraphviz installation, please have a look at the advice here https://github.com/pygraphviz/pygraphviz/issues/100. Or just remove its import, draw_solution function (line 85) and its call on the line 138.
