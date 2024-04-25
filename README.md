# Reinforcement Learning with Human Feedback (RLHF): Bridging the Gap between AI and Human Expertise

This is a simple implementation of RLHF based on the  paper "Learning to summarize from human feedback" for Toronto Metropolitan University , DS8008 Natural Language Processing Course as part of the its Data Science Master's (MSc) program.

# Data

The original raw source of the data used for this experiment comes from Reddit Posts from the below links,

* [Train Dataset]([https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/train.jsonl](https://www.kaggle.com/competitions/rossmann-store-sales/data))

These datasets are downloaded and stored under *datasets/* folder.

# Setup

This project requires **high memory**.The results of this project is obtained by running in Google Colab environment with High Memory.The PyTorch Forecasting library triggered some bugs when ran in GPU ,Hence the models was trained on CPU only.

Below are the steps to run the experiments,

1. Clone this repository to your local machine.
2. Create a conda or virtual environment by executing below command.
   
4. Install the packages/requirements by executing 
5. Go to the root folder and execute the below commands
   ```
   
   ```
   
6. Open the **nlp_rlhf_project.ipynb** file and follow the Instructions.
7. Please note running this notebook will incur cost.(Please budget approx *400-600CAD*) and will take approx 1 day 4 hours to complete the pipeline run based on the current settings. 


# References

1. Learning to summarize from human feedback [link](https://arxiv.org/abs/2009.01325)(*Base Paper*)
2. Secrets of RLHF in Large Language Models, Secrets of RLHF in Large Language Models Part I: PPO [link]( https://arxiv.org/pdf/2307.04964.pdf)
3. Secrets of RLHF in Large Language Models, Part II: Reward Modeling [link](https://arxiv.org/pdf/2401.06080.pdf)
4. Tutorial Reinforment Learning from Human Feedback(*Code Implementation*) [link](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback)
5. Google Cloud RLHF[link](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-rlhf)
6. Wangchunshu Zhou, Ke Xu, "Learning to compare for better training and evaluation of open domain natural language generation models", 2020, [link](https://arxiv.org/pdf/2002.05058.pdf)
7. Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu Tom B. Brown Alec Radford Dario Amodei Paul Christiano Geoffrey Irving, "Fine-tuning language models from human preferences", 2020, [link](https://arxiv.org/pdf/1909.08593.pdf)
   
