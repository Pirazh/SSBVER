from torch import nn

from .utils import trunc_normal_

class SSLHead(nn.Module):
    def __init__(self, in_dim, 
                        out_dim, 
                        use_bn=False, 
                        norm_last_layer=True, 
                        nlayers=3, 
                        hidden_dim=2048, 
                        bottleneck_dim=256):
        super(SSLHead, self).__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, 
                                                            out_dim, 
                                                            bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ReIDHead(nn.Module):
    def __init__(self, num_classes, embed_dim, neck, neck_feat):
        super(ReIDHead, self).__init__()
        
        self.neck = neck
        self.neck_feat = neck_feat
        if self.neck == 'no':
            self.classifier = nn.Linear(embed_dim, num_classes)

        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(embed_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

            self.bottleneck.apply(self.weights_init_kaiming)
        
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        B, C = x.shape

        if self.neck == 'no':
            feat = x
        elif self.neck == 'bnneck':
            feat = self.bottleneck(x)
        
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, x
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    
    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)