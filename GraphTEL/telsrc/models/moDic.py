import numpy as np

motifDict = {
    # 2: 1 type
    2: ["""A -- B"""],
    # 3: 2 type
    3: ["""A -- B B -- C""", """A -- B B -- C C -- A"""],
    # 4: 6 type
    4: ["""A -- B B -- C C -- D""",
        """A -- B B -- C B -- D""",
        """A -- B B -- C C -- D D -- A""",
        """A -- B B -- C C -- D B -- D""",
        """A -- B A -- C B -- D C -- D B -- C""",
        """A -- B B -- C B -- D A -- C A -- D C -- D"""],
}

# Since there are too many motif modes for 5-vertex (21 types) or 6-vertex (112 types),
# Besides, the 5-vertex and 6-vertex motifs are relatively less than 2, 3 and 4-vertex motifs. 
# So we do not choose to directly give all the motif formulas manually.
# We treat a 5/6-vertex motif as a combination of the 3/4-vertex motifs.
# By combining the adj of the 3/4-vertex motifs at node-level, 5/6-vertex motif adj matrix is obtained.

motifADict = {

    3: [np.array([[0, 1, 0], 
                  [1, 0, 1], 
                  [0, 1, 0]]),
        np.array([[0, 1, 1], 
                  [1, 0, 1], 
                  [1, 1, 0]])],
    
    4: [np.array([[0, 1, 0, 0], 
                  [1, 0, 1, 0], 
                  [0, 1, 0, 1], 
                  [0, 0, 1, 0]]),
        np.array([[0, 1, 0, 0], 
                  [1, 0, 1, 1], 
                  [0, 1, 0, 0], 
                  [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 1], 
                  [1, 0, 1, 0], 
                  [0, 1, 0, 1], 
                  [1, 0, 1, 0]]),
        np.array([[0, 1, 0, 0], 
                  [1, 0, 1, 1], 
                  [0, 1, 0, 1], 
                  [0, 1, 1, 0]]),
        np.array([[0, 1, 1, 0], 
                  [1, 0, 1, 1], 
                  [1, 1, 0, 1], 
                  [0, 1, 1, 0]]),
        np.array([[0, 1, 1, 1], 
                  [1, 0, 1, 1], 
                  [1, 1, 0, 1], 
                  [1, 1, 1, 0]]),],
}
