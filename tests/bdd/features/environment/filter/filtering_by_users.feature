Feature: These are the scenarios to filter the dataset by users

  Scenario: test min_consumption if the comparison is greater or greater and equal
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
    When filtered by users with min_consumption
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
      | 2         | 1         | 4.5         | 104           |
      | 2         | 3         | 4.5         | 106           |

  Scenario: test min_consumption when it results in a empty df
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
    And using the minimum of expected ratings as "5"
    When filtered by users with min_consumption
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |

  Scenario: test num_users selection
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 1         | 2         | 3.5         | 101           |
      | 1         | 1         | 4.0         | 102           |
      | 2         | 2         | 4.5         | 103           |
    And using the number of users as "1"
    And random seed equals to "0"
    When filtered by users with num_users
    Then the output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 2         | 2         | 4.5         | 103           |
