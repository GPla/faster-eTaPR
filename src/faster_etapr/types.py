from typing import Any
from typing import NamedTuple

__all__ = [
    'Metric',
    'eTaRecall',
    'eTaPrecision',
]


class Metric(NamedTuple):
    """
    Container for an `eTa` metric. Can be unpacked as a tuple in the order
    below.

    Attributes:
        value [float]: Value of the metric.
        detection_score [float]: Detection score (percentage of counted
          segments).
        portion_score [float]: Portion score (portion of overlap with
          segments).
        segments [int]: Number of segments which contribute to the
          score. For the recall this is the number of detected segments and
          for the precision as the number correct predictions.

    """

    value: float
    detection_score: float
    portion_score: float
    segments: int


class eTaRecall(Metric):
    """
    Container for the ``recall`` of
    `eTa <https://dl.acm.org/doi/10.1145/3477314.3507024>`_. When calling
    `eTaRecall._asdict()` the
    contents will be mapped to:

    .. code-block:: python

      {
        'eta/recall': value,
        'eta/recall_detection': detection_score,
        'eta/recall_portion': portion_score,
        'eta/detected_anomalies': segments,
      }

    Attributes:
      value [float]: Value of the metric.
      detection_score [float]: Detection score (percentage of counted
        segments).
      portion_score [float]: Portion score (portion of overlap with
        segments).
      segments [int]: Number of segments which contribute to the
        score. For the recall this is the number of detected segments and
        for the precision as the number correct predictions.

    """

    def _asdict(self) -> dict[str, Any]:
        return {
            'eta/recall': self.value,
            'eta/recall_detection': self.detection_score,
            'eta/recall_portion': self.portion_score,
            'eta/detected_anomalies': self.segments,
        }


class eTaPrecision(Metric):
    """
    Container for the ``precision`` of
    `eTa <https://dl.acm.org/doi/10.1145/3477314.3507024>`_.  When calling
    `eTaPrecision._asdict()` the contents will be mapped to:

    .. code-block:: python

      {
        'eta/precision': value,
        'eta/precision_detection': detection_score,
        'eta/precision_portion': portion_score,
        'eta/correct_predictions': segments,
      }

    Attributes:
      value [float]: Value of the metric.
      detection_score [float]: Detection score (percentage of counted
        segments).
      portion_score [float]: Portion score (portion of overlap with
        segments).
      segments [int]: Number of segments which contribute to the
        score. For the recall this is the number of detected segments and
        for the precision as the number correct predictions.

    """

    def _asdict(self) -> dict[str, Any]:
        return {
            'eta/precision': self.value,
            'eta/precision_detection': self.detection_score,
            'eta/precision_portion': self.portion_score,
            'eta/correct_predictions': self.segments,
        }
