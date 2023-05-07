## Ventilator-Pressure-Prediction

This is for STAT8017 project in HKU, althogh the project is done, but there is something still valuable to go further...

### Dataset

The dataset is released by Google brain in kaggle: https://www.kaggle.com/competitions/ventilator-pressure-prediction/data

I only used the train.csv which contains over 6 million records. And I have uploaded to my Google Drive.

You can download the dataset by the following command:

```
python main.py --dataset_download
```
By running this command, you will get 7 csv file in total:

* dataset.csv: The original dataset, used for EDA or data analysis

* train.csv: The training set without feature engineering 

* val.csv: The validation set without feature engineering

* test.csv: The testing set without feature engineering

* trainFE.csv, valFE.csv, testFE.csv: The training set, validation set, testing set after feature engineering

The spliting ratio is 2:1:1, and before spliting the dataset, a Robust Scaler is used to scale the data.

### Modeling

#### Baseline--Random Guess

In the part of random guess, we hope our estimates can be as close as possible to the overall distribution characteristics of the training data while introducing certain randomness as well, then use this mode to fit the validation set. For this reason, we designed two parts: Within-breath fit, and Between-breath guess. Within-breath fits specifically examines the timing relationship of samples within a breath\_id, and Between-breath explores the distribution relationship between samples at the same timestamp between different breath\_ids, and we adjusted the weight of the linear combination of the two through the coefficient $\alpha$, which means:

$$
final\_guess = \alpha \times WithinBreath\_{fit} + (1-\alpha) \times BetweenBreath\_guess
$$

**How to get the WithinBreath\_fit?**

First calculate the average value of the corresponding positions of each sequence in the training set to obtain the sample mean sequence

**How to get the BetweenBreath\_guess?**

Between-breath guess is based on an important assumption, if each breath is regarded as a whole sample, they should be independent of each other and generally follow the normal distribution. That is to say, for each breath\_id, the pressure value under the same timestamp is normally distributed.

After fitting, we get the baseline, which is listed below:

|$\alpha$|$R^2$|MSE|MAE|MAPE|SMAPE|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|-0.27|82.93|5.07|0.36|0.39| 
|0.5|0.31|45.18|3.71|0.35|0.27| 
|1|0.50|32.68|3.66|0.45|0.29| 
  

