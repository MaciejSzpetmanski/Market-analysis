import pytest
import prediction_pipeline as pp
import time

@pytest.mark.parametrize(("path", "name", "date", "n"), [
    ("data/^DJI_data_1d.csv", "^DJI", "2024-09-30", 10),
    ("data/^DJI_data_1d.csv", "^DJI", "2024-10-30", 10),
    ("data/^FCHI_data_1d.csv", "^FCHI", "2024-09-30", 10),
])
def test_load_data(path, name, date, n):
    start_time = time.time()
    for i in range(n):
        x = pp.prepare_data_for_prediction(path, name)
        x = pp.merge_vector_with_pred_date(x, date)
    end_time = time.time()
    execution_time = end_time - start_time
    # prepare data in less than 0.5 second
    assert execution_time / n < 0.5
