__version__ = '0.1.1'

__title__ = 'faster-eTaPR'
__description__ = 'Faster implementation of github.com/saurf4ng/eTaPR.'

__author__ = 'Gorden Platz'
__email__ = '36087062+GPla@users.noreply.github.com'

__copyright__ = 'Copyright (c) 2024 Gorden Platz'


from .etapr import evaluate_from_preds, evaluate_from_ranges

__all__ = [
    'evaluate_from_preds',
    'evaluate_from_ranges',
]
