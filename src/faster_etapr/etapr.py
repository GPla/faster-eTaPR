import einops as eo
import mlnext
import numpy as np
import numpy.typing as npt
from mlnext.utils import check_ndim
from mlnext.utils import check_shape

from .types import eTaPrecision
from .types import eTaRecall
from .utils import check_floats

eps = 1e-12

__all__ = [
    'eTaMetrics',
    'evaluate_from_preds',
    'evaluate_from_ranges',
]


class eTaMetrics:
    """Defines the `enhanced time-aware (eTa)
    <https://dl.acm.org/doi/10.1145/3477314.3507024>`_ precision, recall, and
    f1. Moreover, we can also compute the point-wise and
    `point-adjusted <https://arxiv.org/abs/1802.03903>`_ versions.

    **Motivation**: Anomaly detection is a case of binary classification. As
    such, we want to assign each data point a label 0 (normal) and 1
    (anomalous). To measure the effectiveness of a detection method, we can
    calculate the following performance metrics:

    - Recall(RC): How much of anomalies is detected?
    - Precision (PR): How many predictions (for anomalies) concern real
      anomalies?
    - F1: The harmonic mean of recall and precision, which punishes a large
      spread.
    - Segments (SEG): How many anomaly segments are detected?

    For time series data, i.e., a series of observations, we define an anomaly
    as a subsequence within. Naturally, this opens the possibility to two
    different approaches of calculation:

    - point-based: the prediction for each data point is compared to the
      corresponding label
    - range-based: a sequence predictions for an anomaly segment are compared
      against a sequence of labels (i.e., an anomaly)

    In a point-based approach, we can categorize the predictions as follows:

    - true positives (TP): Number of label predictions that are correctly
      identified as anomalous
    - false positive (FP): Number of labels predictions that are wrongly
      classified as anomalous
    - true negatives (TN): Number of label predictions that are correctly
      identified as normal
    - false negatives (FN): Number of label predictions that are wrongly
      classified as normal

    With this in mind, we would calculate the aforementioned performance
    metrics as follows:

    .. math::
        :nowrap:

        \\begin{align*}
        \\mathrm{RC}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
        \\triangleq \\frac{\\mathrm{TP}}{\\mathrm{TP} + \\mathrm{FN}} \\\\

        \\mathrm{PR}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
        \\triangleq \\frac{\\mathrm{TP}}{\\mathrm{TP} + \\mathrm{FP}} \\\\

        \\mathrm{F1}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
        \\triangleq 2 \\frac{\\mathrm{PR}^{\\mathrm{P}} \\cdot
        \\mathrm{RC}^{\\mathrm{P}}}{\\mathrm{PR}^{\\mathrm{P}} +
        \\mathrm{RC}^{\\mathrm{P}}} = \\frac{2 \\mathrm{TP}}{2\\mathrm{TP}
        + \\mathrm{FP} + \\mathrm{FN}}\\\\

        \\mathrm{SEG}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
        \\triangleq
        \\sum_{\\mathbf{A}_i \\in \\mathcal{A}} \\mathbb{1}(
        \\sum_{\\mathbf{P}_j \\in \\mathcal{P}} |\\mathbf{P}_j \\cap
        \\mathbf{A}_i| > 0)
        \\end{align*}

    where :math:`\\mathbf{y}` are the labels and :math:`\\tilde{\\mathbf{y}}`
    the predictions.

    A common variation to the point-wise approach was proposed by
    `Xu et al. <https://arxiv.org/abs/1802.03903>`_ called point-adjust (PA).
    The idea of point-adjust is to mixture of a point- and range-based
    approach: An anomaly segment :math:`A_i` counts as detected if there is at
    least one correct prediction in the segment. This is achieved by adjusting
    the predictions :math:`\\tilde{\\mathbf{y}}` using the labels
    :math:`\\mathbf{y}` as follows:

    .. math::

        \\tilde{y}^{\\mathrm{PA}}_t = \\begin{cases}
            1, & \\text{if $\\tilde{y}_t = 1$ or $\\mathbf{x}_t \\in
            \\mathbf{A}_i$ and $\\underset{\\mathbf{x}_{t'} \\in
            \\mathbf{A}_i}{\\exists} \\tilde{y}_{t'} = 1$} \\\\
            0, & \\text{otherwise.} \\\\
        \\end{cases}

    The following example illustrates the changes that are made to the
    predictions :math:`\\tilde{\\mathbf{y}}` to obtain the adjusted predictions
    :math:`\\tilde{\\mathbf{y}}^\\mathrm{PA}` (the changed positions are
    underlined):

    .. math::
        :nowrap:

        \\begin{align*}
            \\text{labels} \;
            \\mathbf{y}: & [\;0\;0\;1\;1\;1\;0\;0\;1\;1\;0] \\\\

            \\\\

            \\text{predictions} \;
            \\tilde{\\mathbf{y}}: & [\;1\;0\;0\;0\;1\;1\;0\;0\;0\;0] \\\\

            \\text{adjusted} \;
            \\tilde{\\mathbf{y}}^\\mathrm{PA}: &
            [\;1\;0\;\\underline{1}\;\\underline{1}\;1\;1\;0\;0\;0\;0] \\\\
        \\end{align*}

    Afterward, we can calculate the recall, precision, f1, and segment score
    in the same way as before.

    There are several problems with this approach. Mainly, that it
    overestimates the performance and that a higher f1 score does not
    necessarily constitute in a better detection method. For example, a
    detection method which only detects each anomaly segment by a single point
    is scored the same as a method which correctly detects the full segment
    (before the adjustment). For a more detailed discussion see
    `Kim et al. <https://arxiv.org/abs/2109.05257>`_.

    Range-based approaches compare a predicted anomaly segment to a real
    anomaly segment. Thus, it is possible that a prediction :math:`P_j`
    partially overlaps with an anomaly `A_i`. It can be partially a TP and
    partially a FP. eTaPR (enhanced time-aware precision and recall)
    proposed by
    `Hwang et al. <https://dl.acm.org/doi/abs/10.1145/3477314.3507024>`_
    tackles this problem in two ways.

    TODO: finish motivation.

    Attributes:
        preds (list[tuple[int, int]]): Predictions as a list of ranges.
        labels (list[tuple[int, int]]): Labels as a list of ranges.
        theta_p (float, optional): Precision threshold. Only those
              predictions who overlap with at least `theta_p` with a detected
              anomaly are counted as correct. Defaults to 0.5.
        theta_r (float, optional): Recall threshold. Only those anomalies
            which overlap at least `theta_r` with an correct prediction are
            counted as detected.  Defaults to 0.1.
    """

    def __init__(
        self,
        preds: list[tuple[int, int]],
        labels: list[tuple[int, int]],
        *,
        theta_p: float = 0.5,
        theta_r: float = 0.1,
    ):
        check_floats(
            ('theta_p', theta_p),
            ('theta_r', theta_r),
            min=0,
            max=1,
        )

        self.theta_p = theta_p
        self.theta_r = theta_r

        self.preds = np.array(preds)
        self.labels = np.array(labels)

        self._pred_weights = np.sqrt(self.preds[:, 1] + 1 - self.preds[:, 0])
        self._overlap_score_mat_org = self._calculate_overlap_score_mat()
        self._overlap_score_mat = self._overlap_score_mat_org.copy()
        self._labels_max_score = self.labels[:, 1] + 1 - self.labels[:, 0]
        self._preds_max_score = self.preds[:, 1] + 1 - self.preds[:, 0]

        self._pruning()

    def _calculate_overlap_score_mat(self):
        """Calculation of the overlap matrix (n_anomalies, n_predictions).
        A row represents the overlap of predictions with one anomaly.
        If we add the values col-wise (sum(axis=1)), then we get overlap of
        an anomaly (each row) with all predictions. If we add the values
        row-wise (sum(axis=0)), then we get the overlap of a prediction
        (each col) with all anomalies.
        """

        len_l, len_p = len(self.labels), len(self.preds)

        labels_matrix = eo.repeat(self.labels, 'l r -> l p r', p=len_p)
        preds_matrix = eo.repeat(self.preds, 'p r -> l p r', l=len_l)

        detected_starts = np.maximum(
            labels_matrix[..., 0],
            preds_matrix[..., 0],
        )
        detected_ends = np.minimum(
            labels_matrix[..., 1],
            preds_matrix[..., 1],
        )

        overlap_score_mat = np.clip(
            detected_ends + 1 - detected_starts,
            a_min=0,
            a_max=None,
        )

        return overlap_score_mat

    def _pruning(self):
        """Pruning of the overlap matrix. In this process, we eliminate
        rows / cols from the matrix such that only predictions/anomalies remain
        which belong to the set of correct predictions and detected anomalies.
        """

        if len(self.labels) == 0 or len(self.preds) == 0:
            return

        while True:
            labels_portion = self._overlap_score_mat.sum(axis=1) / (
                self._labels_max_score
            )
            label_ids = list(
                set(np.where(labels_portion < self.theta_r)[0])
                - set(np.where(labels_portion == 0.0)[0])
            )
            if label_ids:
                self._overlap_score_mat[label_ids] = np.zeros(
                    (len(label_ids), self._overlap_score_mat.shape[1])
                )

            preds_portion = self._overlap_score_mat.sum(axis=0) / (
                self._preds_max_score
            )

            pred_ids = list(
                set(np.where(preds_portion < self.theta_p)[0])
                - set(np.where(preds_portion == 0.0)[0])
            )
            if pred_ids:
                self._overlap_score_mat[..., pred_ids] = np.zeros(
                    (self._overlap_score_mat.shape[0], len(pred_ids))
                )

            if len(label_ids) == 0 and len(pred_ids) == 0:
                break

    def recall(self) -> eTaRecall:
        """Calculates the `enhanced time-aware recall (eTaR)
        <https://dl.acm.org/doi/10.1145/3477314.3507024>`_. Recall answers the
        question of "How much of anomalies is detected?"

        The recall :math:`\\mathrm{RC}^\\mathrm{eTa}` is calculated as a
        combination of the detection score :math:`s^\\mathrm{RD}` and the
        portion score :math:`s^\\mathrm{RP}` as follows:

        .. math::

            \\mathrm{RC}^{\\mathrm{eTa}}(\\tilde{\\mathbf{y}}, \\mathbf{y})
            \\triangleq
            \\frac{1}{|\\mathcal{A}|}
            \\sum_{A_i \\in \\mathcal{A}}
            \\frac{
                s^{\\mathrm{RD}}(A_i) + s^{\\mathrm{RD}}(A_i)
                \\cdot s^{\\mathrm{RP}}(A_i)
            }{2}

        where :math:`\\tilde{\\mathbf{y}}` are the predictions,
        :math:`\\mathbf{y}` the labels, :math:`A_i` an anomaly, and
        :math:`\\mathcal{A}` the set of all anomalies. The recall
        :math:`\\mathrm{RC}^\\mathrm{eTa}` is the average over all anomaly
        segments :math:`\\mathcal{A}`, but only those anomalies
        :math:`A_i` contribute to the overall score which belong to the set
        of the detected anomalies :math:`\\mathcal{A}^D`. Thus, the recall is
        a measure of how well we can anomaly segments.

        The detection score :math:`s^\\mathrm{RD}` of a anomaly :math:`A_i`
        is defined as:

        .. math::

            s^{\\mathrm{RD}}(A_i) = \\begin{cases}
            1, & \\text{if $A_i \\in \\mathcal{A}^D$}\\\\
            0, & \\text{otherwise},
            \\end{cases}

        where :math:`\\mathcal{A}^D` is the set of detected anomalies. An
        anomaly :math:`A_i` belongs to this set, if the overlapped portion
        with a correct prediction `P_j \\in \\mathcal{P}^C` is greater than
        `theta_r`. Hence, the detection score :math:`s^\\mathrm{RD}` indicates
        whether an anomaly :math:`A_i` is detected or not.

        The portion score :math:`s^\\mathrm{RP}` is the proportion of an
        anomaly :math:`A_i` which intersects with a correct prediction
        :math:`P_j \\in \\mathcal{P}^C`. Mathematically defined as follows,

        .. math::

            s^{\\mathrm{RP}}(\\mathbf{A}_i) =
            \\frac{
                \\sum_{\\mathbf{P}_j \\in \\mathcal{P}^C}
                |\\mathbf{A}_i \\cap \\mathbf{P}_j|
            }{
                |\\mathbf{A}_i|
            }.

        Returns:
            eTaRecall: Returns a namedtuple containing the
              - precision
              - detection score
              - portion score
              - number of correct predictions

        """

        if len(self.labels) == 0 or len(self.preds) == 0:
            return eTaRecall(0.0, 0.0, 0.0, 0)

        rec_portion = self._overlap_score_mat.sum(axis=1) / (
            self._labels_max_score
        )

        detection_scores = np.where(rec_portion >= self.theta_r, 1.0, 0.0)
        detection_score = detection_scores.sum() / len(detection_scores)

        portion_scores = np.clip(rec_portion, a_min=0.0, a_max=1.0)
        portion_score = portion_scores.mean()

        recall = (
            (detection_scores + detection_scores * portion_scores) / 2
        ).mean()

        detected_segments = detection_scores.sum()

        return eTaRecall(
            recall,
            detection_score,
            portion_score,
            detected_segments,
        )

    def precision(self) -> eTaPrecision:
        """Calculates the `enhanced time-aware precision (eTaP)
        <https://dl.acm.org/doi/10.1145/3477314.3507024>`_. Precision
        answers the question of "How many predictions (for anomalies) concern
        real anomalies?".

        The precision :math:`\\mathrm{PR}^\\mathrm{eTa}` is calculated as a
        combination of the detection score :math:`s^\\mathrm{PD}` and the
        portion score :math:`s^\\mathrm{PP}` as follows:

        .. math::

            \\mathrm{PR}^{\\mathrm{eTa}}(\\tilde{\\mathbf{y}}, \\mathbf{y})
            \\triangleq
            \\sum_{P_j \\in \\mathcal{P}} \\left(
                \\frac{s^{\\mathrm{PD}}(P_j) +
                s^{\\mathrm{PD}}(P_j) \\cdot
                s^{\\mathrm{PP}}(P_j)}{2}
            \\right) \\cdot w_{p},

        where :math:`\\tilde{\\mathbf{y}}` are the predictions,
        :math:`\\mathbf{y}` the labels, :math:`P_j` a prediction,
        :math:`\\mathcal{P}` the set of all predictions and :math:`w_{p}` a
        weight for the prediction,

        .. math::

            w_p = \\frac{
                \\sqrt{|P_j|}
            }{
                \\sum_{P_i \\in \mathcal{P}} \sqrt{|P_i|}
            }

        The overall square roots of the lengths of all predictions
        :math:`\sum_{\mathbf{Q} \in \mathcal{P}} \sqrt{|\mathbf{Q}|}` restricts
        the precision score the range [0, 1]. Furthermore, it penalizes the
        detection method for lengthy and frequent incorrect predictions.

        The detection score :math:`s^\\mathrm{PD}` of a prediction :math:`P_j`
        is defined as:

        .. math::

            s^{\\mathrm{PD}}(P_j) = \\begin{cases}
            1, & \\text{if $P_j \\in \\mathcal{P}^C$} \\\\
            0, & \\text{otherwise},
            \\end{cases}

        where :math:`\\mathcal{P}^C` is the set of correct predictions. A
        prediction :math:`P_j` belongs to this set, if at least
        :math:`\\theta_p` of the prediction :math:`P_j` overlaps with a
        detected anomaly :math:`A_i \\in \\mathcal{A}^D`.

        Thus, a prediction :math:`P_j` can only contribute if it is precise
        enough and belongs to the set of correct predictions
        :math:`\\mathcal{P}^C`. Over all predictions :math:`\\mathcal{P}`,
        it is the ratio of correct predictions :math:`\\mathcal{P}^C` to the
        number of all predictions :math:`\\mathcal{P}`, i.e.,
        :math:`\\frac{|\\mathcal{P}^C|}{|\\mathcal{P}|}`.

        The portion score :math:`s^\\mathrm{PP}` is proportion of the
        overlapping parts with a detected anomaly :math:`A_i`:

        .. math::

            s^\\mathrm{PP}(P_j) = \\frac{
                \\sum_{A_i \\in \\mathcal{A}} | A_i \\cap P_j |
            }{
                | P_j |
            }

        Thus, the precision :math:`\\mathrm{PR}^\\mathrm{eTa}` is a measure of
        the quality of the predictions. Only relevant predictions :math:`P_j`,
        i.e., whose overlapping portions are greater than :math:`\\theta_p`,
        can directly contribute to the overall score. However, incorrect
        predictions :math:`P_j \\notin \\mathcal{P}^C` can impact the score
        through the weighting scheme :math:`w_p`.

        Returns:
            eTaPrecision: Returns a namedtuple containing the
              - precision
              - detection score
              - portion score
              - number of correct predictions

        """
        if len(self.labels) == 0 or len(self.preds) == 0:
            return eTaPrecision(0.0, 0.0, 0.0, 0)

        preds_portion = (
            self._overlap_score_mat.sum(axis=0) / self._preds_max_score
        )
        weight = self._pred_weights / self._pred_weights.sum()

        detection_scores = np.where(preds_portion >= self.theta_p, 1.0, 0.0)
        detection_score = (detection_scores * weight).sum()

        portion_scores = np.clip(preds_portion, a_min=0.0, a_max=1.0)
        portion_score = (portion_scores * weight).sum()

        precision = (detection_scores + detection_scores * portion_scores) / 2
        precision = (precision * weight).sum()

        correct_predictions = detection_scores.sum()

        return eTaPrecision(
            precision,
            detection_score,
            portion_score,
            correct_predictions,
        )

    def f1(self, precision: float, recall: float) -> float:
        """Calculates the F1 score from `precision` and `recall` as the
        harmonic mean:

        .. math::

            \\mathrm{F1} \\triangleq 2 \\frac{
                \\mathrm{PR} \\cdot \\mathrm{RC}
            }{
                \\mathrm{PR} + \\mathrm{RC}
            }

        Args:
            precision (float): Precision score.
            recall (float): Recall score.

        Returns:
            float: Returns the F1 score.
        """
        return (2 * precision * recall) / (precision + recall + eps)

    def scores(self) -> dict[str, float | int]:
        """Calculates the enhanced time-aware (eTa) scores. All keys in the
        result mapping are prefixed with ``eta/``.

        Returns:
            dict[str, float | int]: Returns a mapping containing:
              - ``eta/recall``: recall score
              - ``eta/recall_detection``: detection score of the recall
              - ``eta/recall_portion``: portion score of the recall
              - ``eta/detected_anomalies``: number of detected anomalies
              - ``eta/precision``: precision score
              - ``eta/precision_detection``: detection score of the precision
              - ``eta/precision_portion``: portion score of the precision
              - ``eta/correct_predictions``: number of correct predictions
              - ``eta/f1``: f1 score (harmonic mean of precision and recall)
              - ``eta/TP``: number of true positives (points counted)
              - ``eta/FP``: number of false positives (points counted)
              - ``eta/FN``: number of false negatives (points counted)
              - ``eta/wrong_predictions``: number of wrong predictions
              - ``eta/missed_anomalies``: number of undetected anomalies
              - ``eta/anomalies``: total number of anomalies
              - ``eta/segments``: percentage of detected anomalies
        """

        eTaR = self.recall()
        eTaP = self.precision()

        eTaF1 = self.f1(eTaP.value, eTaR.value)

        TP_point = self._overlap_score_mat.sum()
        TP_range = np.count_nonzero(self._overlap_score_mat.sum(axis=0))

        FP_point = self._preds_max_score.sum() - TP_point
        FP_range = len(self.preds) - TP_range

        FN_point = self._labels_max_score.sum() - TP_point
        FN_range = len(self.labels) - TP_range

        anomalies = len(self.labels)
        detected_anomalies = np.count_nonzero(
            self._overlap_score_mat.sum(axis=1)
        )
        segments = detected_anomalies / (anomalies + eps)

        return {
            **eTaR._asdict(),
            **eTaP._asdict(),
            'eta/f1': eTaF1,
            'eta/TP': TP_point,
            'eta/FP': FP_point,
            'eta/FN': FN_point,
            'eta/wrong_predictions': FP_range,
            'eta/missed_anomalies': FN_range,
            'eta/anomalies': anomalies,
            'eta/segments': segments,
        }

    def point_precision(self) -> float:
        """Calculates the point-wise precision score. Precision answers the
        question of "How many predictions (for anomalies) concern real
        anomalies?". In a point-wise manner, we categorize each prediction
        into true positives (TP), false positives (FP), true negative (TN),
        and false negatives (TN). Then we can calculate the precision as
        as ``TP / (TP + FP)``.

        Returns:
            float: Returns the point-wise precision.
        """
        return self._overlap_score_mat_org.sum() / (
            self._preds_max_score.sum() + eps
        )

    def point_recall(self) -> float:
        """Calculates the point-wise recall score. Recall answers the question
        of "How much of anomalies is detected?".In a point-wise manner, we
        categorize each prediction into true positives (TP), false positives
        (FP), true negative (TN), and false negatives (TN). Then we can
        calculate the recall as ``TP / (TP + FN)``.

        Returns:
            float: Returns the point-wise recall.
        """

        return self._overlap_score_mat_org.sum() / (
            self._labels_max_score.sum() + eps
        )

    def point_scores(self) -> dict[str, float | int]:
        """Calculates the point-wise (traditional) scores. Each data point can
        be categorized as either true positive (TP), false positive (FP),
        true negative (TN) or false negative (FN). Then, we can calculate the
        metrics as follows:

        .. math::
            :nowrap:

            \\begin{align*}
            \\mathrm{RC}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
            \\triangleq \\frac{\\mathrm{TP}}{\\mathrm{TP} + \\mathrm{FN}} \\\\

            \\mathrm{PR}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
            \\triangleq \\frac{\\mathrm{TP}}{\\mathrm{TP} + \\mathrm{FP}} \\\\

            \\mathrm{F1}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
            \\triangleq 2 \\frac{\\mathrm{PR}^{\\mathrm{P}} \\cdot
            \\mathrm{RC}^{\\mathrm{P}}}{\\mathrm{PR}^{\\mathrm{P}} +
            \\mathrm{RC}^{\\mathrm{P}}} = \\frac{2 \\mathrm{TP}}{2\\mathrm{TP}
            + \\mathrm{FP} + \\mathrm{FN}}\\\\

            \\mathrm{SEG}^{\\mathrm{P}}(\\tilde{\\mathbf{y}}, \\mathbf{y}) &
            \\triangleq
            \\sum_{\\mathbf{A}_i \\in \\mathcal{A}} \\mathbb{1}(
            \\sum_{\\mathbf{P}_j \\in \\mathcal{P}} |\\mathbf{P}_j \\cap
            \\mathbf{A}_i| > 0)
            \\end{align*}

        All keys in the return mapping are prefixed with ``point/``.

        Returns:
            dict[str, float | int]: Returns a mapping containing:
              - ``point/recall``: point-wise recall (TP / (TP + FN))
              - ``point/precision``: point-wise precision (TP / (TP + FP))
              - ``point/f1``: point-wise f1 score
              - ``point/TP``: number of true positives, correctly classified as 1
              - ``point/FP``: number of false positive, incorrectly classified as 1
              - ``point/FN``: number of false negatives, incorrectly classified as
                0
              - ``point/anomalies``: total number of anomalies
              - ``point/detected_anomalies``: number of detected anomalies (at
                least one point detected)
              - ``point/segments``: percentage of detected anomalies
        """
        recall = self.point_recall()
        precision = self.point_precision()

        f1 = self.f1(precision, recall)

        TP = self._overlap_score_mat_org.sum()
        FP = self._preds_max_score.sum() - TP
        FN = self._labels_max_score.sum() - TP

        anomalies = len(self.labels)
        detected_anomalies = np.where(
            self._overlap_score_mat_org.sum(axis=1) > 0, 1.0, 0
        ).sum()
        segments = detected_anomalies / anomalies

        return {
            'point/recall': recall,
            'point/precision': precision,
            'point/f1': f1,
            'point/TP': TP,
            'point/FP': FP,
            'point/FN': FN,
            'point/anomalies': anomalies,
            'point/detected_anomalies': detected_anomalies,
            'point/segments': segments,
        }

    def point_adjust_precision(self) -> float:
        """Calculates the `point-adjusted <https://arxiv.org/abs/1802.03903>`_
        precision. Precision answers the question of how accurate our
        predictions are. The point-adjusted precision is calculated in the
        same way as the point-wise precision (TP / (TP + FP)). However, the
        predictions are adjusted before calculation using the ground-truth.
        All predictions for an anomaly are set to 1 if at least one correct
        prediction for that anomaly segment exists.

        Returns:
            float: Returns the point-adjust precision.
        """
        TPs = (
            np.clip(self._overlap_score_mat_org.sum(axis=1), 0, 1)
            * self._labels_max_score
        ).sum()

        FPs = self._preds_max_score.sum() - self._overlap_score_mat_org.sum()

        return TPs / (TPs + FPs)

    def point_adjust_recall(self) -> float:
        """Calculates the `point-adjusted <https://arxiv.org/abs/1802.03903>`_
        recall. Recall answers the question of how much of anomaly is detected.
        The point-adjusted recall is calculated in the same way as the
        point-wise recall (TP / (TP + FN)). However, the predictions are
        adjusted before calculation using the ground-truth. All predictions
        for an anomaly are set to 1 if at least one correct
        prediction for that anomaly segment exists.

        Returns:
            float: Reutrns the point-adjusted recall.
        """
        TPs = (
            np.clip(self._overlap_score_mat_org.sum(axis=1), 0, 1)
            * self._labels_max_score
        ).sum()

        return TPs / self._labels_max_score.sum()

    def point_adjust_scores(self) -> dict[str, float]:
        """Calculates the `point-adjusted <https://arxiv.org/abs/1802.03903>`_
        recall, precision, and f1. The metrics are calculated in the same way
        as the point-wise scores but the predictions are adjusted before
        calculation using the ground-truth. All predictions for an anomaly are
        set to 1 if at least one correct prediction for that anomaly segment
        exists.

        Returns:
            dict[str, float]: Returns the point-adjusted scores:
              - ``point_adjust/recall``: point-adjusted recall
              - ``point_adjust/precision``: point-adjusted precision
              - ``point_adjust/f1``: point-adjusted f1
        """
        precision = self.point_adjust_precision()
        recall = self.point_adjust_recall()

        f1 = self.f1(precision, recall)

        return {
            'point_adjust/recall': recall,
            'point_adjust/precision': precision,
            'point_adjust/f1': f1,
        }

    @classmethod
    def from_preds(
        cls,
        y_hat: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        theta_p: float = 0.5,
        theta_r: float = 0.1,
    ) -> 'eTaMetrics':
        """Creates an instance from point-wise predictions and labels.

        Args:
            y_hat (npt.ArrayLike): Predictions (point-wise).
            y (npt.ArrayLike): Labels (point-wise).
            theta_p (float, optional): Precision threshold. Only those
              predictions who overlap with at least `theta_p` with a detected
              anomaly are counted as correct. Defaults to 0.5.
            theta_r (float, optional): Recall threshold. Only those anomalies
              which overlap at least `theta_r` with an correct prediction are
              counted as detected.  Defaults to 0.1.

        Returns:
            eTaMetrics: Returns an instance.
        """

        y, y_hat = np.squeeze(y), np.squeeze(y_hat)
        check_ndim(y, y_hat, ndim=1)
        check_shape(y, y_hat)

        preds = mlnext.find_anomalies(y_hat)
        labels = mlnext.find_anomalies(y)

        eta = eTaMetrics(preds, labels, theta_p=theta_p, theta_r=theta_r)

        return eta


def evaluate_from_preds(
    y_hat: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict[str, float | int]:
    """Calculates the `enhanced time-aware (eTa)
    <https://dl.acm.org/doi/10.1145/3477314.3507024>`_, point-wise, and
    `point-adjusted <https://arxiv.org/abs/1802.03903>`_ performance
    metrics (and some other miscellaneous metrics). To see how these
    metrics are calculated, check out the respective methods in
    :class:`.eTaMetrics`.

    Args:
        y_hat (npt.ArrayLike): Predictions (point-wise).
        y (npt.ArrayLike): Labels (point-wise).
        theta_p (float, optional): Precision threshold. Only those
          predictions who overlap with at least `theta_p` with a detected
          anomaly are counted as correct. Defaults to 0.5.
        theta_r (float, optional): Recall threshold. Only those anomalies
          which overlap at least `theta_r` with an correct prediction are
          counted as detected.  Defaults to 0.1.

    Returns:
        dict[str, float | int]: Returns a mapping with all metrics:
          - ``eta/recall``: eTa recall score
          - ``eta/recall_detection``: detection score of the recall
          - ``eta/recall_portion``: portion score of the recall
          - ``eta/detected_anomalies``: number of detected anomalies
          - ``eta/precision``: eTa precision score
          - ``eta/precision_detection``: detection score of the precision
          - ``eta/precision_portion``: portion score of the precision
          - ``eta/correct_predictions``: number of correct predictions
          - ``eta/f1``: f1 score (harmonic mean of precision and recall)
          - ``eta/TP``: number of true positives (points counted)
          - ``eta/FP``: number of false positives (points counted)
          - ``eta/FN``: number of false negatives (points counted)
          - ``eta/wrong_predictions``: number of wrong predictions
          - ``eta/missed_anomalies``: number of undetected anomalies
          - ``eta/anomalies``: total number of anomalies
          - ``eta/segments``: percentage of detected anomalies
          - ``point/recall``: point-wise recall (TP / (TP + FN))
          - ``point/precision``: point-wise precision (TP / (TP + FP))
          - ``point/f1``: point-wise f1 score
          - ``point/TP``: number of true positives, correctly classified as 1
          - ``point/FP``: number of false positive, incorrectly classified as 1
          - ``point/FN``: number of false negatives, incorrectly classified as
            0
          - ``point/anomalies``: total number of anomalies
          - ``point/detected_anomalies``: number of detected anomalies (at
            least one point detected)
          - ``point/segments``: percentage of detected anomalies
          - ``point_adjust/recall``: point-adjusted recall
          - ``point_adjust/precision``: point-adjusted precision
          - ``point_adjust/f1``: point-adjusted f1

    Example:

        >>> import faster_etapr
        >>> faster_etapr.evaluate_from_ranges(
        ...     y_hat=[0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        ...     y=    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        ...     theta_p=0.5,
        ...     theta_r=0.1,
        ... )
        {
            'eta/recall': 0.3875,
            'eta/recall_detection': 0.5,
            'eta/recall_portion': 0.275,
            'eta/detected_anomalies': 2.0,
            'eta/precision': 0.46476766302377037,
            'eta/precision_detection': 0.46476766302377037,
            'eta/precision_portion': 0.46476766302377037,
            'eta/correct_predictions': 2.0,
            'eta/f1': 0.4226312395393011,
            'eta/TP': 4,
            'eta/FP': 5,
            'eta/FN': 7,
            'eta/wrong_predictions': 2,
            'eta/missed_anomalies': 2,
            'eta/anomalies': 4,
            'eta/segments': 0.499999999999875,
            'point/recall': 0.45454545454541323,
            'point/precision': 0.5555555555554939,
            'point/f1': 0.49999999999945494,
            'point/TP': 5,
            'point/FP': 4,
            'point/FN': 6,
            'point/anomalies': 4,
            'point/detected_anomalies': 3.0,
            'point/segments': 0.75,
            'point_adjust/recall': 0.9090909090909091,
            'point_adjust/precision': 0.7142857142857143,
            'point_adjust/f1': 0.7999999999995071
        }

    """
    eta = eTaMetrics.from_preds(
        y_hat=y_hat,
        y=y,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    return {
        **eta.scores(),
        **eta.point_scores(),
        **eta.point_adjust_scores(),
    }


def evaluate_from_ranges(
    preds: list[tuple[int, int]],
    labels: list[tuple[int, int]],
    *,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict[str, float | int]:
    """Calculates the `enhanced time-aware (eTa)
    <https://dl.acm.org/doi/10.1145/3477314.3507024>`_, point-wise, and
    `point-adjusted <https://arxiv.org/abs/1802.03903>`_ performance
    metrics (and some other miscellaneous metrics). To see how these
    metrics are calculated, check out the respective methods in
    :class:`.eTaMetrics`.

    Args:
        y_hat (list[tuple[int, int]]): Predictions as list of ranges.
        y (list[tuple[int, int]]): Labels as list of ranges.
        theta_p (float, optional): Precision threshold. Only those
          predictions who overlap with at least `theta_p` with a detected
          anomaly are counted as correct. Defaults to 0.5.
        theta_r (float, optional): Recall threshold. Only those anomalies
          which overlap at least `theta_r` with an correct prediction are
          counted as detected.  Defaults to 0.1.

    Returns:
        dict[str, float | int]: Returns a mapping with all metrics:
          - ``eta/recall``: eTa recall score
          - ``eta/recall_detection``: detection score of the recall
          - ``eta/recall_portion``: portion score of the recall
          - ``eta/detected_anomalies``: number of detected anomalies
          - ``eta/precision``: eTa precision score
          - ``eta/precision_detection``: detection score of the precision
          - ``eta/precision_portion``: portion score of the precision
          - ``eta/correct_predictions``: number of correct predictions
          - ``eta/f1``: f1 score (harmonic mean of precision and recall)
          - ``eta/TP``: number of true positives (points counted)
          - ``eta/FP``: number of false positives (points counted)
          - ``eta/FN``: number of false negatives (points counted)
          - ``eta/wrong_predictions``: number of wrong predictions
          - ``eta/missed_anomalies``: number of undetected anomalies
          - ``eta/anomalies``: total number of anomalies
          - ``eta/segments``: percentage of detected anomalies
          - ``point/recall``: point-wise recall (TP / (TP + FN))
          - ``point/precision``: point-wise precision (TP / (TP + FP))
          - ``point/f1``: point-wise f1 score
          - ``point/TP``: number of true positives, correctly classified as 1
          - ``point/FP``: number of false positive, incorrectly classified as 1
          - ``point/FN``: number of false negatives, incorrectly classified as
            0
          - ``point/anomalies``: total number of anomalies
          - ``point/detected_anomalies``: number of detected anomalies (at
            least one point detected)
          - ``point/segments``: percentage of detected anomalies
          - ``point_adjust/recall``: point-adjusted recall
          - ``point_adjust/precision``: point-adjusted precision
          - ``point_adjust/f1``: point-adjusted f1

    Example:

        >>> import faster_etapr
        >>> faster_etapr.evaluate_from_ranges(
        ...     y_hat=[(1, 1), (3, 4), (7, 9), (11, 13)],
        ...     y=    [(1, 2), (5, 7), (10, 14), (16, 16)],
        ...     theta_p=0.5,
        ...     theta_r=0.1,
        ... )
        {
            'eta/recall': 0.3875,
            'eta/recall_detection': 0.5,
            'eta/recall_portion': 0.275,
            'eta/detected_anomalies': 2.0,
            'eta/precision': 0.46476766302377037,
            'eta/precision_detection': 0.46476766302377037,
            'eta/precision_portion': 0.46476766302377037,
            'eta/correct_predictions': 2.0,
            'eta/f1': 0.4226312395393011,
            'eta/TP': 4,
            'eta/FP': 5,
            'eta/FN': 7,
            'eta/wrong_predictions': 2,
            'eta/missed_anomalies': 2,
            'eta/anomalies': 4,
            'eta/segments': 0.499999999999875,
            'point/recall': 0.45454545454541323,
            'point/precision': 0.5555555555554939,
            'point/f1': 0.49999999999945494,
            'point/TP': 5,
            'point/FP': 4,
            'point/FN': 6,
            'point/anomalies': 4,
            'point/detected_anomalies': 3.0,
            'point/segments': 0.75,
            'point_adjust/recall': 0.9090909090909091,
            'point_adjust/precision': 0.7142857142857143,
            'point_adjust/f1': 0.7999999999995071
        }

    """
    eta = eTaMetrics(
        preds=preds,
        labels=labels,
        theta_p=theta_p,
        theta_r=theta_r,
    )

    return {
        **eta.scores(),
        **eta.point_scores(),
        **eta.point_adjust_scores(),
    }
