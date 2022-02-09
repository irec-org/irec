# Metrics

The recommender metrics supported by IREC are listed below.

| Metric | Reference | Description
| :---: | --- | :--- |
| [Hits](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L183) | [Link](link) | Number of recommendations made successfully. 
| [Precision](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L139) | [Link](link) | Precision is defined as the percentage of predictions we get right.
| [Recall](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L97) | [Link](link) | Represents the probability that a relevant item will be selected.  
| [EPC](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L232) | [Link](link) | It represents the novelty for each user and it is measured by the expected number of seen relevant recommended items not previously seen.  
| [EPD](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L359) | [Link](link) | EPD is a distance-based novelty measure, which looks at distances between the items inthe user’s profile and the recommended items. 
| [ILD](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L312) | [Link](link) | It represents the diversity between the list of items recommended. This diversity is measure by the Pearson correlation of the item’s features vector. 
| [Gini Coefficient](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L449)| [Link](link) | Diversity is represented as the Gini coefficient – a measure of distributional inequality. It is measured as the inverse of cumulative frequency that each item is recommended.
| [Users Coverage](https://github.com/irec-org/irec/blob/24a28734f757e95d1423dac4ada9dfb85fa05b73/irec/metrics.py#L498) | [Link](link) | It represents the percentage of distinctusers that are interested in at least k items recommended (k ≥ 1).
