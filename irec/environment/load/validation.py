# TODO
# yaml file:
# 'MovieLens 100k TrainTest':
#   TrainTestLoader:
#     dataset:
#       train:
#         path: ./data/datasets/MovieLens 100k/
#         file_delimiter: ","
#         skip_head: true
#       test:
#         path: ./data/datasets/MovieLens 100k/
#         file_delimiter: ","
#         skip_head: true
#       validation:
#         path: ./data/datasets/MovieLens 100k/
#         validation_size: 0.1

# TODO: if validation["path"] exists, we will read from this file
#       if not, we should create the validation

# In the other case:
# 'MovieLens 100k':
#   DefaultLoader:
#     dataset:
#       path: ./data/datasets/MovieLens 100k/
#       random_seed: 0
#       file_delimiter: ","
#       skip_head: true
#     prefiltering: #optional filters
#       filter_users:
#         min_consumption: 50
#         num_users: 50
#       filter_items:
#         min_ratings: 1
#         num_items: 50
#     splitting:
#       strategy: random
#       train_size: 0.8
#       test_consumes: 5
#     validation:
#       validation_size: 0.1