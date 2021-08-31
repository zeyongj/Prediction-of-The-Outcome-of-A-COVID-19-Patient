# Milestone 3

## Introudction
- SFU CMPT 459: Data Mining, Group Project: Prediction of The Outcome of A COVID-19 Patient.
- In the second and third milestones, the aim is to build models and using this model to predict the outcomes on cases_test.csv dataset.
- Running main.py may take **several days**.


## Libraries

- numpy
- pandas
- sklearn
- warnings
- seaborn
- datetime
- lightgbm
- math
- pickle

# Execution
- If using `main.py`:
	- First you need to create new data and models folders.
	- Then you need to manually upload `case_train_processed.csv` and `cases_test_processed.csv` to the data folder, upload `lgb_classifier.pkl`, `mlp_classifier.pkl` and `svc_classifier.pkl` in the last milestone into models folder. 
	- Last just command: `python src/main.py`. It should run automatically, but the time complexity is extremely high. The total running time may exceed 1 day.
- If using `main.ipynb`: 
	- First you need to create new data, plots, models and results folders respectively in the left file bar. 
	- Then you need to manually upload `case_train_processed.csv` and `cases_test_processed.csv` to the data folder and `lgb_classifier.pkl`, `mlp_classifier.pkl` and `svc_classifier.pkl` in the last milestone into models folder.
	- And now you can press `Ctrl + F9` on the colab page, it should run one unit by one unit automatically. But due to the extremely high time complexity, the running time may exceed 1 day.
	
## License

This work is licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (or any later version). 

`SPDX-License-Identifier: Apache-2.0-or-later`

## Disclaimer

This repository is ONLY for backup. Students should NEVER use this repository to finish their works, IN ANY WAY.

It is expected that within this course, the highest standards of academic integrity will be maintained, in
keeping with SFU’s Policy S10.01, `Code of Academic Integrity and Good Conduct`.

In this class, collaboration is encouraged for in-class exercises and the team components of the assignments, as well
as task preparation for group discussions. However, individual work should be completed by the person
who submits it. Any work that is independent work of the submitter should be clearly cited to make its
source clear. All referenced work in reports and presentations must be appropriately cited, to include
websites, as well as figures and graphs in presentations. If there are any questions whatsoever, feel free
to contact the course instructor about any possible grey areas.

Some examples of unacceptable behaviour:
- Handing in assignments/exercises that are not 100% your own work (in design, implementation,
wording, etc.), without a clear/visible citation of the source.
- Using another student's work as a template or reference for completing your own work.
- Using any unpermitted resources during an exam.
- Looking at, or attempting to look at, another student's answer during an exam.
- Submitting work that has been submitted before, for any course at any institution.

All instances of academic dishonesty will be dealt with severely and according to SFU policy. This means
that Student Services will be notified, and they will record the dishonesty in the student's file. Students
are strongly encouraged to review SFU’s Code of Academic Integrity and Good Conduct (S10.01) available
online at: http://www.sfu.ca/policies/gazette/student/s10-01.html.

## Author

Zeyong Jin

April 26th, 2021

