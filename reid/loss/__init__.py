from __future__ import absolute_import

from .tripletSGG import TripletLoss
from .lsr import LSRLoss
from .triplet import TripletLoss
# from .lsr import LSRLoss
from .neightboor import InvNet
from .neightboor_office import InvNet_office

__all__ = [
    'TripletLoss',
    'LSRLoss',
    'InvNet',
    'InvNet_office', 
    'InvNet_office'
]
