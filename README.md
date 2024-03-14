# Sentiment Analysis of Twitter Users Towards the Israel Palestine Conflict

This project employs sentiment analysis and machine learning models to gauge English-speaking Twitter users'
sentiments toward the Israel-Palestine conflict. Utilizing the Octoparse API, tweets were scraped under hashtags
such as #IsraelPalestineConflict, #ProIsrael, #ProPalestine, etc. from October 7, 2023 (the date of the Hamas attack),
onwards. The final dataset comprises of 1,360 rows including polarity, date posted, user, tweet content, and likes.
The analysis involves comparing Roberta and Vader polarity models and using machine learning (Decision Trees,
Support Vector Machines, Naïve Bayes) to categorize tweets into pro-Israel, pro-Palestine, or neutral. From there,
the performances were analyzed through metrics like accuracy, precision, and recall on the testing data. The Roberta
model outperformed Vader, revealing predominantly negative sentiments. Machine learning models achieved an
accuracy of ~53%, with insights indicating a majority of tweets leaning towards Pro-Palestine. Ethical
considerations include privacy in data scraping and responsible result presentation. Future directions include model
refinement and dataset expansion.

## Introduction
<p>In the age of social media, the daily data influx is a valuable resource for gauging public opinions on diverse topics.
Motivated by applying in-class machine learning concepts to a current real-world issue, the project delves into the
Israel-Palestine conflict. The focus is on capturing and analyzing public sentiment expressed by English-speaking
Twitter users, recognizing the limitations of traditional methods such as opinion polls in providing nuanced insights.
The project seeks to address this gap by offering a more accurate and effective alternative to traditional polls,
leveraging sentiment analysis and machine learning models. The study used Twitter data from October 7, 2023,
employing models such as Vader, Roberta, Support Vector Machines (SVM), Decision Tree (DT), and Naïve Bayes
(NB). The findings reveal a prevailing pro-Palestine sentiment, highlighting the challenges of sentiment
classification on Twitter. Through model refinement, we anticipate contributing to a more nuanced understanding of
public sentiment and evaluating the potential impact.</p>


## Related Work
<p>Other projects around this topic include sentiment analysis studies around other topics. The paper ‘Sentiment
Analysis on Twitter Data’ conducted a general sentiment analysis study on Twitter data using machine learning
algorithms. The study categorizes tweets into positive, negative, or neutral sentiments related to specific query
terms, enabling applications to assess customer feedback for product improvement (Sahayak et al., 2015).
Additionally, a project specifically analyzed Twitter tweets from December 2022 to January 2023 about ChatGPT
(Huang, 2023). Three research questions were asked regarding the main topics and sentiments of the conversations
around early ChatGPT users. The study allowed us to understand and assess ChatGPT's capability, effectiveness, and
challenges. A third study that was found to be the most similar to our project was ‘Sentiment Analysis of Political
Tweets for Israel using Machine Learning’ (Gangwar et al., 2022). The creators analyzed tweets in May 2021 and
scraped data using hashtags #IsraelUnderAttack, #IStandWithIsrael, #WeStandWithIsrael, and
#IsraelPalestineConflict. They utilized SVM, DT, and NB with the NB model having the highest accuracy of
93.21%. However, upon inspection, some errors in their code led to the accuracy being higher than expected. </p>

## Data
<p>To better understand public opinion on the Israel Palestine situation, we decided to focus on Twitter sentiment by
scraping and analyzing tweets as our data set. We believed that Twitter provides real time opinions on current news,
and since this is a recently popular topic there would be highly active opinions on the situation. In total we scraped
1,360 tweets. We found that the highest sentiment count was Pro Palestine at 600, followed by Neutral at 410, then
Pro Israel at 350. For hashtags, #ProPalestine came up to be 451, #ProIsrael at 397, and #IsraelPalestineWar at 231.</p>

## Approach
<p>The methodological framework for this project involved a systematic four-step process: overall sentiment analysis,
feature engineering, training/testing, and the comparison of model metrics. In the initial stage, natural language
processing was applied to determine the collective sentiment of Twitter users. This analysis involved computing the
score of each tweet based on the hashtag by which it was categorized. The Vader model assessed the compound
score, while the RoBERTa model provided a breakdown of sentiment categories for each hashtag-associated tweet.
Moving to the second stage, team members manually classified tweets into Pro-Israel, neutral, or Pro-Palestine
categories, creating a new "polarity" feature. In the third stage, Decision Trees, Support Vector Machines (SVMs),
and Naive Bayes were employed to train and test the models using this new feature. Finally, accuracy and precision
scores were compared across the machine learning models to determine the best-performing one.</p>

# Experiments
The techniques used for sentiment analysis are as follows:

# Vader Model

<p>VADER (Valence Aware Dictionary and Sentiment Reasoner) is a natural language processing model used for
sentiment analysis. It assigns sentiment scores (positive, negative, or neutral) to words and phrases, considering
context and grammar rules. Vader utilizes a sentiment lexicon and a set of grammatical rules to capture sentiments,
making it suitable for analyzing the emotional tone of the text in various contexts. (Amanmyrat Abdullayev, 2022) </p>

# Roberta Model
<p> RoBERTa (Robustly optimized BERT approach) is another natural language processing model used for semantic
analysis. Roberta excels in capturing contextualized representations of words due to it being trained on a Twitter
dataset, making it highly effective for tasks like sentiment analysis and text classification. Its training approach
including an increased attention to textual context contributes to a more advanced performance compared to the
Vader model. The Roberta model similarly provides scores in positive, negative, and neutral categories. (Amanmyrat
Abdullayev, 2022) </p>

# Support Vector Classifiers
<p>Support Vector Classifiers are supervised machine learning algorithms designed for classification and regression.
Their goal is to find a hyperplane in a high-dimensional space that maximizes the margin between different classes
of data points. Support vectors, the data points closest to the hyperplane, play a crucial role in determining the
optimal separation. Once the optimal hyperplane is identified, SVM can efficiently classify new data points by
determining on which side of the hyperplane they fall. SVMs are effective in moderate-sized datasets, offering
robustness against overfitting, but can become computationally expensive. (1.4. Support Vector Machines, 2023) </p>

# Decision Trees
<p>Decision Trees are a machine learning algorithm also used for both classification and regression. They have a
flowchart-like structure where each internal node denotes a decision based on a feature, each branch signifies the
outcome of that decision, and each leaf node represents the final predicted label or value. The goal is to recursively
split the data into subsets based on the most informative features, creating a hierarchical set of decision rules. (1.10.
Decision Trees, 2023)</p>

# Naive Bayes
<p>Naive Bayes is a probabilistic classification algorithm. It is based on Bayes' theorem, which calculates the
probability of a hypothesis given observed evidence. The "naive" assumption in Naive Bayes is that features used
for classification are conditionally independent, meaning the presence or absence of one feature does not affect the
presence or absence of another. The algorithm calculates the probability of each class given the input features and
selects the class with the highest probability as the predicted class for a given instance. </p>
