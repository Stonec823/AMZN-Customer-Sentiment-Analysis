
# PureView AI: An Unbiased NLP Tool For Customer Sentiment

This is team 184's repository to build a tableau dashboard with sentiment and topic modeling allowing users to filter on product attributes, review sentiment, review topic and text contained within user reviews. 

## Data 
https://amazon-reviews-2023.github.io/


# Setup 

Go to GCP Vertex AI > Workbench > Managed notebooks https://console.cloud.google.com/vertex-ai/workbench/instances

Select create new at the top to create an instance, don't sweat too much about the details here, just select create. This step may take some time to load, don't worry just be a bit patient. Once you have an instance, click open jupyterlab.

On the ribbon above, there is a 'Git' click on that then 'Clone a repository' and paste the link to this repo https://github.gatech.edu/jtriemer3/CSE-6242-Amazon-Review-Sentiment.git 
You'll have to login to your gatech github here, mine is jtriemer3 then my password.

Next open a terminal and run the below commands

```
source ~/.bashrc
```
#### Create env from yml file - may take some time to download
By using this yml environment we can be sure that we all are sharing dependencies and can run the same code on the same python version 

```
conda env create -f CSE-6242-Amazon-Review-Sentiment/team184-env.yml
```
This may take a bit as it installs all the necessary packages.
```
conda activate team184-env
```

NOTE - **Whenever you pip install something new, like pandas, you will need to export the env again (see code below) to overwrite the yml so that we all can share that new package. If you skip this step, the code may not run for everyone due to dependency issues.**

```
conda env export > team184-env.yml
```

#### Creating the conda kernel
This essentially will allow us to use our conda env can be used in a jupyter notebook
```
DL_ANACONDA_ENV_HOME="${DL_ANACONDA_HOME}/envs/team184-env"
python -m ipykernel install --prefix "${DL_ANACONDA_ENV_HOME}" --name team184-env --display-name team184
```

# Once the environment is setup, you should be able to run through our process
1. Run 1_text_preprocessing.py
2. Run 2_topic_processing.py
3. Run 3_topic_processing.py

