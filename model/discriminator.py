import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
from torch.nn.utils import spectral_norm


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def squash(input_tensor, dim=-1, epsilon=1e-12):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a PatchGAN discriminator
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [snconv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                snconv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            snconv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [snconv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        sequence += [snconv2d(ndf * nf_mult, ndf, kernel_size=1, stride=2, padding=1)]
        sequence += [snconv2d(ndf, 128, kernel_size=1, stride=1, padding=0)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)  # return size: [bs, 128, 8, 8]


class PrimaryCapsLayer(nn.Module):
    def __init__(self, cap_dim=8):
        super(PrimaryCapsLayer, self).__init__()
        self.cap_dim = cap_dim

    def forward(self, x):
        outputs = x.permute(0, 2, 3, 1).contiguous()         # [bs, 8, 8, 128=16*8]
        outputs = outputs.view(x.size(0), -1, self.cap_dim)  # [bs, 1024=8*8*16, 8]
        outputs = squash(outputs)                            # [bs, 1024, 8]
        return outputs


class DigitCapsLayer(nn.Module):
    def __init__(self, args, num_digit_cap=1, num_prim_cap=1024, in_cap_dim=8, out_cap_dim=16, num_iterations=3):
        super(DigitCapsLayer, self).__init__()
        self.num_prim_cap = num_prim_cap
        self.num_digit_cap = num_digit_cap
        self.num_iterations = num_iterations
        self.device = args.device
        # [1, 1024, 16, 8]
        self.W = nn.Parameter(0.01 * torch.randn(num_digit_cap, num_prim_cap, out_cap_dim, in_cap_dim))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        b = torch.zeros(x.size(0), self.num_digit_cap, self.num_prim_cap, device=self.device)  # [bs, 10, 1024]
        for i in range(0, self.num_iterations):
            c = F.softmax(b, dim=-1)
            if i == self.num_iterations - 1:
                v = squash(torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))
                return torch.squeeze(v, dim=-2)  # [bs, 1, 512]
            else:
                v = squash(torch.sum(c[:, :, :, None] * temp_u_hat, dim=-2, keepdim=True))
                b += torch.sum(v * temp_u_hat, dim=-1)


class Decoder(nn.Module):
    def __init__(self, args, input_width=64, input_height=64, input_channel=1, n_cls=3, out_dim=30):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.classes = n_cls
        self.device = args.device
        self.reconstraction_layers = nn.Sequential(
            snlinear(out_dim * n_cls, 512),
            nn.ReLU(inplace=True),
            snlinear(512, 1024),
            nn.ReLU(inplace=True),
            snlinear(1024, 5120),
            nn.ReLU(inplace=True),
            snlinear(5120, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = F.one_hot(max_length_indices, self.classes)
        masked = masked.to(self.device)
        t = (x * masked[:, :, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


def define_D(args, in_channels, img_size, n_classes, output_dim):
    net = netD(args=args, in_channels=in_channels, img_size=img_size, n_classes=n_classes, output_dim=output_dim)

    def init_fn(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    return net.apply(init_fn)


class netD(nn.Module):
    def __init__(self, args, in_channels, img_size, n_classes, output_dim):
        super(netD, self).__init__()
        self.classes = n_classes
        self.in_channels = in_channels
        self.img_size = img_size

        # patch CNN
        self.patch_cnn = NLayerDiscriminator(input_nc=in_channels)
        # primary capsule layer
        self.primary_capsules = PrimaryCapsLayer()
        # decoder
        self.decoder = Decoder(args, img_size, img_size, in_channels, n_classes, output_dim)
        # digital capsule layer
        self.aux_layer = DigitCapsLayer(args=args, num_digit_cap=n_classes, out_cap_dim=output_dim)
        self.adv_layer = DigitCapsLayer(args=args, num_digit_cap=1, out_cap_dim=output_dim)

    def forward(self, img):
        x = self.patch_cnn(img)
        x = self.primary_capsules(x)
        aux_out = self.aux_layer(x)
        adv_out = self.adv_layer(x)

        validity = torch.sqrt((adv_out ** 2).sum(dim=2))

        reconstructions, masked = self.decoder(aux_out)

        aux_out = torch.sqrt((aux_out ** 2).sum(dim=2))

        return validity, aux_out, reconstructions, masked

    def loss_fn(self, data, x, target, reconstructions, use_rec=True):
        if use_rec:
            return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)
        else:
            return self.margin_loss(x, target)

    def margin_loss(self, v_c, labels, one_hot=True):
        if one_hot:
            # Convert labels to one-hot form
            labels = F.one_hot(labels, self.classes)

        batch_size = v_c.size(0)

        left = F.relu(0.9 - v_c, inplace=True).view(batch_size, -1)
        right = F.relu(v_c - 0.1, inplace=True).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()
        return loss

    def reconstruction_loss(self, data, reconstructions):
        mse_loss = nn.MSELoss()
        loss = mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005


class Discriminator(nn.Module):
    def __init__(self, args, in_channels, img_size, n_classes, output_dim):
        super(Discriminator, self).__init__()
        self.netD = define_D(args, in_channels, img_size, n_classes, output_dim)

    def forward(self, x):
        return self.netD(x)


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device
    # dis = Discriminator(args, 1, 128, 3, 32)
    # dis = dis.to(device)
    # # print(dis)
    # input = torch.rand((2, 1, 128, 128)).to(args.device)
    # out = dis(input)
    #
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)

    model = NLayerDiscriminator(input_nc=128)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))