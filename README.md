# PureView AI: An Unbiased NLP Tool For Customer Sentiment

This is team 184's repository to build a tableau dashboard with sentiment and topic modeling allowing users to filter on product attributes, review sentiment, review topic and text contained within user reviews. 

Link to the dashboard here! -
https://public.tableau.com/app/profile/oswaldo.ceballos/viz/PureViewAI/Dashboard1?publish=yes


## Data -
For this project, we will use data from the below link for raw customer reviews for topic and sentiment modelling-
https://amazon-reviews-2023.github.io/


# Setup -
Download the zip and extract it, then open the new directory you extracted to in a terminal, then follow the below directions


#### Create env from yml file - may take some time to download
By using this yml environment we can be sure that we all are sharing dependencies and can run the same code on the same python version 

```
conda env create -f CSE-6242-Amazon-Review-Sentiment/team184-env.yml
```

This may take a bit as it installs all the necessary packages. Be sure you cd to the right directory.

```
conda activate team184-env
```

# To run the pureview ETL process, run the below command:
```
python CSE-6242-Amazon-Review-Sentiment/pureview_local.py
```
