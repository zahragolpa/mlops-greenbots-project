import os
from deepchecks.tabular.suites import model_evaluation
from helpers.utils import create_dc_dataset


def validate_model(train_ds, test_ds, model, model_name="model"):
	print(f"Validating model {model_name} with deepchecks...")
	train_ds_dc = create_dc_dataset(train_ds)
	test_ds_dc = create_dc_dataset(test_ds)
	evaluation_suite = model_evaluation()
	suite_result = evaluation_suite.run(train_ds_dc, test_ds_dc, model)

	if not os.path.exists('deepchecks_reports/model_validation'):
		os.makedirs('deepchecks_reports/model_validation')

	report_path = suite_result.save_as_html(f'deepchecks_reports/model_validation/{model_name}.html')
	print(f'Saved the deepchecks model validation report at {report_path}')
	return suite_result.passed(fail_if_warning=False, fail_if_check_not_run=False)

