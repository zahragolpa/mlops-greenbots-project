import os
from deepchecks.tabular.suites import data_integrity, train_test_validation
from helpers.utils import create_dc_dataset


def validate_single_dataframe(dataframe, dataframe_name="dataframe"):
	print(f"Validating dataframe {dataframe_name} with deepchecks...")
	dc_dataset = create_dc_dataset(dataframe)
	suite = data_integrity()
	if dataframe.shape[0] == 1:
		suite = suite.remove(0)
	suite_result = suite.run(dc_dataset)

	data_integrity_report_dir = 'deepchecks_reports/data_integrity_validation'
	if not os.path.exists(data_integrity_report_dir):
		os.makedirs(data_integrity_report_dir)

	report_path = suite_result.save_as_html(f'{os.path.join(data_integrity_report_dir, dataframe_name)}.html')
	print(f'Saved the deepchecks data integrity validation report at {report_path}')
	return suite_result.passed(fail_if_warning=False, fail_if_check_not_run=False)


def validate_train_test_dataframe(train_df, test_df, dataframe_name="dataframe"):
	print(f"Validating train and test dataframes {dataframe_name} with deepchecks...")
	dc_train = create_dc_dataset(train_df)
	dc_test = create_dc_dataset(test_df)
	suite = train_test_validation()
	suite_result = suite.run(train_dataset=dc_train, test_dataset=dc_test)

	train_test_report_dir = 'deepchecks_reports/train_test_data_validation'
	if not os.path.exists(train_test_report_dir):
		os.makedirs(train_test_report_dir)

	report_path = suite_result.save_as_html(f'{os.path.join(train_test_report_dir, dataframe_name)}.html')
	print(f'Saved the deepchecks data validation report at {report_path}')
	return suite_result.passed(fail_if_warning=False, fail_if_check_not_run=False)
