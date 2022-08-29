#!/bin/bash

#=============================================== NSGA-II (Evol_algo) ========================================================#

python3 search_algo/main_search.py --problem hw_nas --n-init-sample 2 --pop-init-method lhs --pop-size 50 --nb-gen 2 \
--search-resume True --offspring False ;

exit 0