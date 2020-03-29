# this script is for defining model

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticReg( nn.Module ):

    def __init__( self, features, classes ):
        
        super( LogisticReg, self ).__init__()
        
        self.features = features
        self.classes  = classes
        
        self.lm1 = nn.Linear( features, 64, bias=True )
        self.lm2 = nn.Linear( 64, classes, bias=True )

    # still can define other tasks
    def forward( self, x ):
        x = F.relu( self.lm1( x ) )
        x = F.relu( self.lm2( x ) )
        return x
