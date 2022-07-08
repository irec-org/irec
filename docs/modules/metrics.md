# Metrics

This module contains all evaluation metrics available in the iRec. Its goal is to provide distinct options to be selected during the previous setup. These metrics are suitable to the recommendation scenario and are usually split into a few groups: Accuracy Coverage, Novelty and Diversity. In our architecture, they follow an implementation pattern where each metric has two methods: 

(1) compute, in which the entire calculation is performed for a given user;

(2) update, which updates the historic of items in each user during the interactive scenario. 

The [recommender metrics](https://github.com/irec-org/irec/tree/update-info/irec/offline_experiments/metrics) supported by iRec are listed below.


| Metric | Reference | Description
| :---: | --- | :--- |
| [Hits](irec/offline_experiments/metrics/hits.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Number of recommendations made successfully. 
| [Precision](irec/offline_experiments/metrics/precision.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Precision is defined as the percentage of predictions we get right.
| [Recall](irec/offline_experiments/metrics/recall.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Represents the probability that a relevant item will be selected.  
| [EPC](irec/offline_experiments/metrics/epc.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | Represents the novelty for each user and it is measured by the expected number of seen relevant recommended items not previously seen.  
| [EPD](irec/offline_experiments/metrics/epd.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | EPD is a distance-based novelty measure, which looks at distances between the items in the user’s profile and the recommended items. 
| [ILD](irec/offline_experiments/metrics/ild.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | It represents the diversity between the list of items recommended. This diversity is measured by the Pearson correlation of the item’s features vector. 
| [Gini Coefficient](irec/offline_experiments/metrics/gini_coefficient_inv.py)| [Link](https://dl.acm.org/doi/abs/10.1145/3298689.3347040?casa_token=-QId0RoJsHgAAAAA:er_vhmem2f1h-_Yv4YJ3E0vXg6F-0tnu62c08l4g_9_TFmNDUEpBJTZQZgUniyH1fhEhkcWVUBWGPl8) | Diversity is represented as the Gini coefficient – a measure of distributional inequality. It is measured as the inverse of cumulative frequency that each item is recommended.
| [Users Coverage](irec/offline_experiments/metrics/users_coverage.py) | [Link](https://link.springer.com/article/10.1007/s13042-017-0762-9) | It represents the percentage of distinct users that are interested in at least k items recommended (k ≥ 1).