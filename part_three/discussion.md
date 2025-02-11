### Given the two sets above, how would you create a truth set of matching products with minimal manual effort?

Under the assumption that both the dealing with the same language i.e. Spanish.

* Method 1: Do an exact match on the columns `name` in the provided .csv file with the `product_name` column in the Open Food Facts dataset. Clean the text before exact matching i.e. either all in lowercase and remove special characters.
* Method 2: Use an embedding model to generate vectors for `name` and `product_name` for the two column, respectively. For each product from dataset A then do a similarity search with those in dataset B to find match with the highest score.
  * If the dataset is very large do an approximate nearest neighbour.

Using the above two methods, if we get a manageable number of matched pairs, we can then manually validate if the matches are correct for certain.

### There are situations when negative cases are not available. Imagine you only have a small set of known positives matches that has been manually validated. Explain briefly how you would deal with that type of scenario to create negative examples.

Products belonging to same or similar category are likely to have more similarity than products belonging to different categories.

* Concatenate the text from category and product_name columns of the both the datasets
* Vectorize the concatenated text using an embedding mode (TF_IDF/LLM-based)
* Cluster the vectors (ex: k-means)
* Vectors (products) belong to the same cluster are similar. Those belong to different cluster are dissimilar.
* To generate negative pairs of data; pick a product from cluster X, belonging to dataset A. Pick a product from cluster Y, belonging to dataset B.

### If you found a method to obtain negative cases, how do you distinguish between easy and hard samples?

Since the method suggested is based on clustering, I look at the L2 distance.

* Combine the text of category with product_name and vectorize the text.
* Then compare the L2 distance to distinguish between easy and difficult matches.

### Given the above, what split strategy would you follow? Once the model is deployed, can you describe how you would use new labeled data?

* Ensure both easy and hard negative are present in training and test sets in similar ratios (shuffle the dataset before splitting into train and test set).
* If the new labeled data has the different distribution compared to those used in the training the deployed model, the model needs to be retrained and this retrained model needs to be then deployed.

### For the task of automatically matching pairs of description, explain at high-level what typeof model, loss, and KPIs would you use and why.

Model: BERT

Loss: triplet loss

KPIs: Precision, recall, F1

### Imagine that you develop a system to retrieve similar descriptions to a provided one. Would you use the same configuration as in the previous question?

Partially similar, i.e. using an embedding model to generate vectors for the descriptions text. The second part would involve a retriever rather than a matcher.  Instead of returning one matching descriptions (top-1), the model should now return the (top-k) matching documents. Here we could look into different metrics like top-k precision/recall where the aim is the best matching descriptions should up the top-k retrieved results
