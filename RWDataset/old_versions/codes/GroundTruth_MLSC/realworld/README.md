# Ground Truth Generation
to compute the ground truth for each node, can use local subgraph counting:
1. [SCOPE](https://github.com/magic62442/subgraph-counting?tab=readme-ov-file), output non-induced count for 5-7 nodes patterns
2. [EVOKE](https://bitbucket.org/nojan-p/orbit-counting/src/master/), output non-induced and induced(need to be converted from non-induced) count for 3-5 nodes patterns
3. [JESSE](https://github.com/biointec/jesse), no k constraint but their implementation only supports <=5 nodes patterns
4. [ORCA](https://file.biolab.si/biolab/supp/orca/#download), output induced count for 3-5 nodes patterns, can use matrices in EVOKE to convert to non-induced
5. GRAPH MATCH, no k constraint but too slow