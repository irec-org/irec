Feature: These are the scenarios to filter the dataset by items

  Scenario: test min_ratings if the comparison is greater or greater and equal
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
      | 2         | 1         | 4.5         | 104           |
      | 3         | 1         | 4.5         | 105           |
      | 2         | 3         | 4.5         | 106           |
    And using the minimum of expected ratings as "2"
    When filtered by items with min_ratings
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
      | 2         | 1         | 4.5         | 104           |
      | 3         | 1         | 4.5         | 105           |

  Scenario: test min_ratings when it results in a empty df
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
    And using the minimum of expected ratings as "5"
    When filtered by items with min_ratings
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |

  Scenario: test num_items selection
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
    And using the number of items as "1"
    And random seed equals to "0"
    When filtered by items with num_items
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 1         | 4.0         | 102           |
