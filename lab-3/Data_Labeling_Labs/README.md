# Modifications

## Lab 1: spam tutorial (01_spam_tutorial.ipynb)
[Lab 1: Data Labeling Spam - Updated Code Link](01_spam_tutorial.ipynb)

* **Focused on four key spam types**:
  1. Emoji spam (`😜`, `😂`, `❤️`)
  2. Phone/email scams (phone numbers, email addresses)
  3. Crypto/investment/money/prize scams (`free`, `win`, `cash`, `prize`, `crypto`, `bitcoin`)
  4. Clickbait / shortened links (`bit.ly`, `tinyurl`, `goo.gl`)

* **Labeling Functions (LFs) updated** to detect these spam types programmatically.

* **Snorkel LabelModel** applied to generate probabilistic labels using the new LFs.

* **Slice-based evaluation** added:
  * Accuracy computed for each spam type separately.
  * Allows insight into which spam types are easier or harder for the model to classify.

* **Model comparison**: Logistic Regression vs LinearSVC.

* Maintained baseline LFs for **short messages** and **excessive exclamations**.

## Lab 02 – Spam Data Augmentation (02_spam_data_augmentation_tutorial.ipynb)
[Lab 2: Spam Data Augmentation - Updated Code Link](02_spam_data_augmentation_tutorial.ipynb)


* **Introduced Synthetic Data Augmentation**  
  - Added new samples by appending spammy phrases or emojis (“🔥 Limited offer!”, “💰 Don’t miss out!”).  
  - Created a balanced dataset combining **original** and **augmented** samples with a new flag: `is_augmented`.

* **Focused on Four Key Spam Types (carried over from Lab 1)**  
  1. Emoji spam 😜😂❤️  
  2. Phone/email scams (numbers, email patterns)  
  3. Crypto/money/prize scams (`free`, `win`, `cash`, `bitcoin`)  
  4. Clickbait / shortened links (`bit.ly`, `tinyurl`, `goo.gl`)

* **Extended Labeling Functions (LFs)**  
  - Reused Lab 1 functions and refined them to detect more nuanced spam variants.  
  - Reapplied **Snorkel’s LabelModel** on the augmented dataset to generate probabilistic labels.

* **Dual-Model Evaluation**  
  - Compared **Logistic Regression** vs **LinearSVC** on test accuracy.  
  - Observed how augmentation affected generalization and precision across categories.

* **Slice-Based Evaluation**  
  - Calculated per-type accuracy for `emoji`, `money`, `clickbait`, and `phone/email` slices.  
  - Identified categories with performance boosts from augmentation.

## Lab 03 – Spam Data Slicing (03_spam_data_slicing_tutorial.ipynb)

[Lab 3: Spam Data Slicing - Updated Code Link](03_spam_data_slicing_tutorial.ipynb)

* Loaded dataset and applied updated Snorkel labeling functions.

* Introduced adaptive LFs focusing on multiple real-world spam patterns.

* Applied Snorkel LabelModel to aggregate weak labels and generate probabilistic training data.

* Trained and compared two models — Logistic Regression and LinearSVC.

* Conducted slice-based evaluation for better interpretability of spam detection results.



---
# The Significance of Data Labeling and the Role of Snorkel in Enhancing Machine Learning

In the realm of machine learning (ML) and artificial intelligence (AI), data is the bedrock upon which models are built and refined. However, raw data, while abundant, often lacks the structure and organization necessary for effective learning. This is where data labeling comes into play. Data labeling is the process of annotating or tagging data points with relevant metadata or labels, providing context and meaning to otherwise unstructured information. This process is crucial for numerous reasons, and its importance cannot be overstated in the development and deployment of ML models.

- First and foremost, labeled data serves as the training material for ML algorithms. Supervised learning, one of the most prevalent paradigms in ML, relies heavily on labeled datasets to learn the relationship between input features and target outputs. Without accurately labeled data, ML models would struggle to generalize patterns and make accurate predictions. Thus, the quality and accuracy of labels directly influence the performance and reliability of ML systems.

- Furthermore, data labeling facilitates the creation of ground truth datasets, which serve as benchmarks for evaluating model performance. By comparing model predictions against accurately labeled data, ML engineers can assess the efficacy of their algorithms and identify areas for improvement. This iterative feedback loop is essential for refining ML models and enhancing their robustness in real-world scenarios.

- Additionally, labeled data enables domain-specific insights and knowledge extraction. By categorizing data points into meaningful classes or attributes, organizations can derive valuable insights about customer preferences, market trends, and business operations. These insights can inform strategic decision-making and drive innovation across various industries, from healthcare to finance to e-commerce.

Despite its undeniable importance, data labeling can be a labor-intensive and time-consuming process, especially when dealing with large volumes of unstructured data. Manual labeling by human annotators is often prone to errors, inconsistencies, and biases, leading to suboptimal model performance. This is where automated labeling tools like Snorkel come into play.

`Snorkel is a powerful framework designed to streamline the data labeling pipeline and mitigate the challenges associated with manual annotation. By leveraging weak supervision techniques, Snorkel enables ML teams to programmatically generate labels from noisy or imperfect sources, such as heuristics, rules, or distant supervision. This approach not only accelerates the labeling process but also improves label quality by aggregating information from multiple sources and learning from noisy signals.`

Moreover, Snorkel provides a unified platform for managing the entire data labeling workflow, from data ingestion to model training. Its flexible and extensible architecture allows users to customize labeling functions, integrate with existing ML pipelines, and adapt to evolving data requirements. By automating tedious labeling tasks and empowering ML engineers to focus on high-level model design and optimization, Snorkel enhances productivity and accelerates the pace of innovation in ML research and development.

In this toturail we will learn fundamental aspects of Snorkel using 3 examples provided by Snorkel team.

# Spam Tutorials
We consider a canonical machine learning problem: classifying spam. This directory contains three tutorials, described below:
* `01_spam_tutorial`: This tutorial dives deep into how we can create, analyze, and use labeling functions for the spam classification task.
* `02_spam_data_augmentation_tutorial`: This tutorial demonstrates how to write, combine and apply transformation functions for performing data augmentation.
* `03_spam_data_slicing_tutorial`: This tutorial shows how we can use slicing functions to identify important slices of the data, for monitoring and improved performance.

