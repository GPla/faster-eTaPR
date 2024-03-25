__version__ = '0.1.0'

__title__ = 'eTaPR'
__description__ = 'Faster implementation of the eTaPR package.'

__author__ = 'Gorden Platz'
__email__ = '36087062+GPla@users.noreply.github.com'

__copyright__ = 'Copyright (c) 2024 Gorden Platz'


from .etapr import calculate_from_preds, calculate_from_ranges

__all__ = [
    'calculate_from_preds',
    'calculate_from_ranges',
]
