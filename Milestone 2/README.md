# Milestone 2

## Introudction
- SFU CMPT 459: Data Mining, Group Project: Prediction of The Outcome of A COVID-19 Patient.
- In the second and third milestones, the aim is to build models and using this model to predict the outcomes on cases_test.csv dataset.
- Running main.py may take several minutes.
- Ensure that main.py and eda.ipynb are running under the src folder otherwise the results and plots folder cannot maintain the correct structure
- When running, the data folder should have the original files.

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

## References

- https://blog.csdn.net/qq_35679701/article/details/107239487
- https://blog.csdn.net/weixin_44132035/article/details/102807785
- https://blog.csdn.net/swordtraveller/article/details/92786837
- https://blog.csdn.net/wlx19970505/article/details/80301193

# Execution
- If using `main.py`, then just command: `python src/main.py`.
- If using main.ipynb: 
	- Create new data, plots and models folders respectively in the left file bar. 
	- Manually upload `case_train_processed.csv` to the data folder and `lgb_classifier.pkl`, `mlp_classifier.pkl` and `svc_classifier.pkl` into models folder.
	- Now you can just click the run button one by one to run it.

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

March 22nd, 2021
