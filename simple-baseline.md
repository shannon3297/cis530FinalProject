# Description of Simple Baseline
Our simple baseline is a majority class baseline that determines the majority class (negative, neutral, or positive sentiment) from the training data and guesses that class for each item in the test set. 

# Sample Output
The sample training file contains 62,307 rows of Tweets text, Tweet ID, place, creation time, user location, username, follower count, retweet count, favorite count, hashtags, and sentiment. This file contains 26,817, 18,090, and 17,346 instances with neutral, positive, and negative sentiment respectively. One sample instance from this training set is:

*   **text**: Wuhan has been in complete quarantine for over 8 weeks. People here are still going about their daily lives as normal. This is pure fantasy. https://t.co/AUZbGNRDjM
*   **id**: 1240727808005070000
*   **place**: ""
*   **created_at**: Thu Mar 19 19:52:15 +0000 2020
*   **user_location**: ""
*   **user_name**: Jamie
*   **followers_count**: 3199
*   **retweet_count**: 1
*   **favorite_count**: 12
*   **hashtags**: ""
*   **sentiment**: positive

Given this sample training file, the test file that contains 15,580 unlabeled instances gets labeled with "neutral" for each instance since neutral is the majority class according to the training set. 

# Test Set Results Evaluated with Scoring Script
When the simple baseline is run on the test set as described above, the scoring scipt outputs the following F1 scores:

*   **Negative**: 0.000
*   **Neutral**: 0.591
*   **Positive**: 0.000
*   **Weighted Average**: 0.248

