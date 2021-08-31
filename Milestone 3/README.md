This is the final milestone of the course project of CMPT 459 E100: Data Mining, finished by Yuqing Wu, Zeyong Jin and Zixi Bai. And the professor is Dr. Martin Ester.

In order to to execute the codes, the following libraries are needed:
numpy
pandas
sklearn
warnings
seaborn
datetime
lightgbm
math
pickle

Please check the following guides to run the codes:
1. If using main.py:
	First you need to create new data and models folders.
	Then you need to manually upload case_train_processed.csv and cases_test_processed.csv to the data folder, upload lgb_classifier.pkl, mlp_classifier.pkl and svc_classifier.pkl in the last milestone into models folder. 
	Last just command: python src/main.py. It should run automatically, but the time complexity is extremely high. The total running time may exceed 1 day.
2. If using main.ipynb: 
	First you need to create new data, plots, models and results folders respectively in the left file bar. 
	Then you need to manually upload case_train_processed.csv and cases_test_processed.csv to the data folder and lgb_classifier.pkl, mlp_classifier.pkl and svc_classifier.pkl in the last milestone into models folder.
	And now you can press Ctrl + F9 on the colab page, it should run one unit by one unit automatically. But due to the extremely high time complexity, the running time may exceed 1 day.

Thanks!
April 26, 2021
