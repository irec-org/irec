Feature: These are the scenarios to filter the dataset globally

  Scenario: test global users selection
    Given a dataset containing
      | int    | int    | float  | int       |
      | userId | itemId | rating | timestamp |
      | 0      | 1      | 3.5    | 101       |
      | 0      | 0      | 4.0    | 102       |
      | 3      | 1      | 4.5    | 103       |
      | 1      | 0      | 4.5    | 104       |
      | 2      | 0      | 4.5    | 105       |
      | 1      | 2      | 4.5    | 106       |
    And using the train size as "0.8"
    And using the tests consumes as "1"
    And random seed equals to "0"
    When split globally
    Then the train output should be equal to
      | int    | int    | float  | int       |
      | userId | itemId | rating | timestamp |
      | 0      | 1      | 3.5    | 101       |
      | 0      | 0      | 4.0    | 102       |
      | 3      | 1      | 4.5    | 103       |
      | 1      | 0      | 4.5    | 104       |
    And the test output should be equal to
      | int    | int    | float  | int       |
      | userId | itemId | rating | timestamp |
      | 2      | 0      | 4.5    | 105       |

  Scenario: test keep all users in global split
    Given a dataset containing
      | int    | int    | float  | int       |
      | userId | itemId | rating | timestamp |
      | 0      | 1      | 3.5    | 101       |
      | 0      | 0      | 4.0    | 102       |
      | 3      | 1      | 4.5    | 103       |
      | 1      | 0      | 4.5    | 105       |
      | 2      | 0      | 4.5    | 105       |
      | 1      | 2      | 4.5    | 106       |
    And using the train size as "0.8"
    And using the tests consumes as "1"
    And random seed equals to "0"
    When split globally
    Then the total users should be equal to 4
