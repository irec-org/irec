Feature: These are the scenarios for filtering the dataset based on a percentage of the user's history

  Scenario: test user_history
    Given a dataset containing
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 0         | 1         | 3.5         | 101           |
      | 0         | 0         | 4.0         | 102           |
      | 0         | 2         | 4.0         | 103           |
      | 1         | 1         | 4.5         | 104           |
      | 1         | 0         | 4.5         | 105           |
      | 1         | 2         | 4.5         | 106           |
      | 2         | 1         | 4.5         | 107           |
      | 2         | 0         | 4.5         | 108           |
    And using the train size as "0.8"
    And using the tests consumes as "1"
    And random seed equals to "0"
    When split user_history
    Then the train output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 0         | 1         | 3.5         | 101           |
      | 0         | 0         | 4.0         | 102           |
      | 1         | 1         | 4.5         | 104           |
      | 1         | 0         | 4.5         | 105           |
      | 2         | 1         | 4.5         | 107           |
    And the test output should be equal to
      | int       | int       | float       | int           |
      | userId    | itemId    | rating      | timestamp     |
      | 0         | 2         | 4.0         | 103           |
      | 1         | 2         | 4.5         | 106           |
      | 2         | 0         | 4.5         | 108           |
