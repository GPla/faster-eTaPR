from itertools import product

import eTaPR_pkg
import mlnext
import numpy as np
import pytest
from etapr.etapr import eTa

rng = np.random.default_rng(1337)
y_hat_y = [
    (
        [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    ),
    (
        rng.integers(0, 2, size=1_000),
        rng.integers(0, 2, size=1_000),
    ),
]
thetas = [
    (theta_p, theta_r)
    for theta_p in np.linspace(0.1, 1, 10)
    for theta_r in np.linspace(0.1, 1, 10)
]

SetupType = tuple[eTa, eTaPR_pkg.eTaPR]


@pytest.fixture(params=list(product(y_hat_y, thetas)), scope='module')
def setup(request) -> SetupType:
    (y, y_hat), (theta_p, theta_r) = request.param

    etapr_new = eTa.from_preds(
        y_hat,
        y,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    etapr_old = eTaPR_pkg.eTaPR(theta_p, theta_r)
    anomalies = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y, 0, 1, True)
    preds = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y_hat, 0, 1, True)
    etapr_old.set(anomalies, preds)

    return etapr_new, etapr_old


@pytest.mark.parametrize('y_hat,y', y_hat_y)
@pytest.mark.parametrize('theta_p,theta_r', thetas)
def test_scores(y_hat, y, theta_p, theta_r):

    etapr_new = eTa.from_preds(
        y_hat,
        y,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    result = etapr_new.scores()

    anomalies = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y, 0, 1, True)
    preds = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y_hat, 0, 1, True)
    exp = eTaPR_pkg.evaluate_w_ranges(anomalies, preds, theta_p, theta_r)

    compare = {
        'eta/precision': 'eTaP',
        'eta/precision_portion': 'eTaPp',
        'eta/precision_detection': 'eTaPd',
        'eta/recall': 'eTaR',
        'eta/recall_portion': 'eTaRp',
        'eta/recall_detection': 'eTaRd',
        # 'eta/wrong_predictions': 'N False Alarm',
        'eta/anomalies': 'n_anomalies',
        'eta/segments': 'segments',
    }

    for k, v in compare.items():
        np.testing.assert_almost_equal(
            act := result[k],
            des := exp[v],
            err_msg=f'{k}: {act} != {des} :{v}',
        )

    len_compare = {
        'eta/detected_anomalies': 'Detected_Anomalies',
        'eta/correct_predictions': 'Correct_Predictions',
    }

    for k, v in len_compare.items():
        np.testing.assert_almost_equal(
            act := result[k],
            des := len(exp[v]),
            err_msg=f'{k}: {act} != {des} :{v}',
        )


@pytest.mark.parametrize('y_hat,y', y_hat_y)
@pytest.mark.parametrize('theta_p,theta_r', thetas)
def test_point_scores(y_hat, y, theta_p, theta_r):

    etapr_new = eTa.from_preds(
        y_hat,
        y,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    result = etapr_new.point_scores()

    exp = mlnext.score.eval_metrics(np.array(y), np.array(y_hat))

    compare = {
        'point/precision': 'precision',
        'point/recall': 'recall',
        'point/f1': 'f1',
        'point/segments': 'anomalies',
    }

    for k, v in compare.items():
        np.testing.assert_almost_equal(
            act := result[k],
            des := exp[v],
            err_msg=f'{k}: {act} != {des} :{v}',
        )

    # anomalies = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y, 0, 1, True)
    # preds = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y_hat, 0, 1, True)
    # exp = eTaPR_pkg.evaluate_w_ranges(anomalies, preds, theta_p, theta_r)

    # compare = {
    #     'point/precision': 'precision',
    #     'point/recall': 'recall',
    # }

    # for k, v in compare.items():
    #     np.testing.assert_almost_equal(
    #         act := result[k],
    #         des := exp[v],
    #         err_msg=f'{k}: {act} != {des} :{v}',
    #     )


@pytest.mark.parametrize('y_hat,y', y_hat_y)
@pytest.mark.parametrize('theta_p,theta_r', thetas)
def test_point_adjust_scores(y_hat, y, theta_p, theta_r):

    etapr_new = eTa.from_preds(
        y_hat,
        y,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    result = etapr_new.point_adjust_scores()

    y = np.array(y)
    y_hat_adjust = mlnext.apply_point_adjust(y_hat=np.array(y_hat), y=y)
    exp = mlnext.score.eval_metrics(y, y_hat_adjust)

    compare = {
        'point_adjust/precision': 'precision',
        'point_adjust/recall': 'recall',
        'point_adjust/f1': 'f1',
    }

    for k, v in compare.items():
        np.testing.assert_almost_equal(
            act := result[k],
            des := exp[v],
            err_msg=f'{k}: {act} != {des} :{v}',
        )

    # anomalies = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y, 0, 1, True)
    # preds = eTaPR_pkg.DataManage.File_IO.load_stream_2_range(y_hat, 0, 1, True)
    # exp = eTaPR_pkg.evaluate_w_ranges(anomalies, preds, theta_p, theta_r)

    # compare = {
    #     'point_adjust/precision': 'point_adjust_precision',
    #     'point_adjust/recall': 'point_adjust_recall',
    #     'point_adjust/f1': 'point_adjust_f1',
    # }

    # for k, v in compare.items():
    #     np.testing.assert_almost_equal(
    #         act := result[k],
    #         des := exp[v],
    #         err_msg=f'{k}: {act} != {des} :{v}',
    #     )


def test_overlap_score_mat(setup: SetupType):
    etapr_new, etapr_old = setup

    result = etapr_new._overlap_score_mat_org
    exp = etapr_old._overlap_score_mat_org

    np.testing.assert_equal(result, exp)


def test_pruning(setup: SetupType):
    etapr_new, etapr_old = setup

    result = etapr_new._overlap_score_mat
    exp = etapr_old._overlap_score_mat_elm

    np.testing.assert_equal(result, exp)


def test_eTaR(setup: SetupType):

    etapr_new, etapr_old = setup

    r_recall, r_detection_score, r_portion_score, r_segments = (
        etapr_new.recall()
    )

    e_recall = etapr_old.eTaR()
    e_detection_score, e_segments = etapr_old.eTaR_d()
    e_portion_score = etapr_old.eTaR_p()

    assert r_recall == e_recall, f'{r_recall} != {e_recall}'
    assert (
        r_detection_score == e_detection_score
    ), f'{r_detection_score} != {e_detection_score}'
    assert (
        r_portion_score == e_portion_score
    ), f'{r_portion_score} != {r_portion_score}'
    assert r_segments == len(e_segments), f'{r_segments} != {e_segments}'


def test_eTaP(setup: SetupType):

    etapr_new, etapr_old = setup

    r_precision, r_detection_score, r_portion_score, r_segments = (
        etapr_new.precision()
    )

    e_precision = etapr_old.eTaP()
    e_detection_score, e_segments = etapr_old.eTaP_d()
    e_portion_score = etapr_old.eTaP_p()

    np.testing.assert_almost_equal(r_precision, e_precision)
    np.testing.assert_almost_equal(r_detection_score, e_detection_score)
    np.testing.assert_almost_equal(r_portion_score, e_portion_score)
    assert r_segments == len(e_segments), f'{r_segments} != {e_segments}'
