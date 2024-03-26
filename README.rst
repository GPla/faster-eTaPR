faster-eTaPR
============

|docs| |pre-commit| |mypy|

.. |docs| image:: https://readthedocs.org/projects/faster-etapr/badge/?version=latest
.. _docs: https://faster-etapr.readthedocs.io/en/latest/?badge=latest

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
.. _pre-commit: https://github.com/pre-commit/pre-commit

.. |mypy| image:: http://www.mypy-lang.org/static/mypy_badge.svg
.. _mypy: http://mypy-lang.org/


Faster implementation (`~200x <#benchmark>`_) of the enhanced time-aware precision and recall (eTaPR) from  `Hwang et al <https://dl.acm.org/doi/10.1145/3477314.3507024>`_.
The original implementation is `saurf4ng/eTaPR <https://github.com/saurf4ng/eTaPR>`_ and this implementation is fully tested against it.

Motivation
----------

The motivation behind the `eTaPR <https://dl.acm.org/doi/10.1145/3477314.3507024>`_ is that it is enough for a detection method to partially detect an anomaly segment, as along as an human expert can find the anomaly around this prediction.
The following illustration (a recreation from the `paper <https://dl.acm.org/doi/10.1145/3477314.3507024>`_) highlights the four cases which are considered by eTaPR:

.. image:: /img/motivation.png
    :width: 80%
    :align: center
    :alt: motivation behind eTaPR

1. A *successful* detection: A human expert can likely find the anomaly :math:`A_1` based on the prediction :math:`P_1`.
2. A *failed* detection: Only a small portion of the prediction :math:`P_2` overlaps with the anomaly :math:`A_2`.
3. A *failed* detection: Most of the prediction :math:`P_3` lies in the range of non-anomalous behavior (prediction starts too early). A human expert will likely regard the prediction :math:`P_3` as incorrect or a false alarm. The prediction :math:`P_3` is *too imprecise* and the anomaly :math:`A_3` is likely to be missed.
4. A *failed* prediction: The prediction :math:`P_4` mostly overlaps with the anomaly :math:`A_4`, but covers only a small portion of the actual anomaly segment. Thus, a human expert is likely to dismiss the prediction :math:`P_4` as incorrect because the full extend of the anomaly remains hidden. The prediction `P_4` contains *insufficient* information about the anomaly.

Note that for case 4, we could still mark the anomaly as detected, if there were more predictions which overlap with the anomaly :math:`A_4`.
Specifically, the handling of the cases 3 and 4 is what sets eTaPR apart from other scoring methods.

If you want an in-depth explanation of the calculation, check out the `documentation <https://faster-etapr.readthedocs.io/>`_.

Getting Started
---------------

Until this package is released on PyPI, you can install it directly from Github, using `pip <https://github.com/pypa/pip>`_ or `uv <https://github.com/astral-sh/uv>`_:

.. code::

    pip install git+https://github.com/GPla/faster-eTaPR.git

.. code::

    uv pip install git+https://github.com/GPla/faster-eTaPR.git

Now, you run your evaluation in python:

.. code::

    import faster_etapr
    faster_etapr.evaluate_from_ranges(
        y_hat=[0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        y=    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        theta_p=0.5,
        theta_r=0.1,
    )
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

We calculate three types of metrics:

- the `enhanced time-aware (eTa)
  <https://dl.acm.org/doi/10.1145/3477314.3507024>`_ metrics under
  ``eta/``
- the (traditional) point-wise metrics under ``point/``
- the `point-adjusted <https://arxiv.org/abs/1802.03903>`_ metrics under
  ``point_adjust/``


.. _benchmark:

Benchmark
---------

A little benchmark with randomly generated inputs (:code:`np.random.randint(0, 2, size=size)`):

+---------+-----------+--------------+--------+
| size    | eTaPR_pkg | faster_etapr | factor |
+=========+===========+==============+========+
| 1 000   | 0.4090    | 0.0032       | ~125x  |
+---------+-----------+--------------+--------+
| 10 000  | 35.8264   | 0.1810       | ~198x  |
+---------+-----------+--------------+--------+
| 20 000  | 148.2670  | 0.6547       | ~226x  |
+---------+-----------+--------------+--------+
| 100 000 | too long  | 55.04712     |        |
+---------+-----------+--------------+--------+

TODO
----

- Upload to PyPI
- Github CI
