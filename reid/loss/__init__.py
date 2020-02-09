from __future__ import absolute_import

<<<<<<< HEAD
from .tripletSGG import TripletLoss
from .lsr import LSRLoss
=======
from .triplet import TripletLoss
# from .lsr import LSRLoss
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
from .neightboor import InvNet
from .neightboor_office import InvNet_office

__all__ = [
    'TripletLoss',
    'LSRLoss',
    'InvNet',
<<<<<<< HEAD
    'InvNet_office', 
=======
    'InvNet_office'
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
]
