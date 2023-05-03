import torch
from torch_utils import misc

class StyleGAN_Wrapper(torch.nn.Module):

    def __init__(self, G):
        super(StyleGAN_Wrapper, self).__init__()
        self.G = G
        self.syn = G.synthesis
        self.mapping = G.mapping

    def forward(self, ws=None, f_latents=None, f_layer=0, mode="wp"):

        if ws is None and f_latents is None:
            return torch.zeros(0,0,0,0).cuda()

        if mode == "wp":
            return self.forward_wp(ws)
        elif mode == "from_f":
            return self.forward_from_f(ws, f_latents, f_layer)
        elif mode == "to_f":
            return self.forward_to_f(ws, f_layer)

    def forward_wp(self, ws, **layer_kwargs):
        misc.assert_shape(ws, [None, self.syn.num_ws, self.syn.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.syn.input(ws[0])
        for name, w in zip(self.syn.layer_names, ws[1:]):
            x = getattr(self.syn, name)(x, w, **layer_kwargs)
        if self.syn.output_scale != 1:
            x = x * self.syn.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.syn.img_channels, self.syn.img_resolution, self.syn.img_resolution])
        x = x.to(torch.float32)
        return x

    def forward_from_f(self, ws, f_latents, f_layer, **layer_kwargs):
        misc.assert_shape(ws, [None, self.syn.num_ws, self.syn.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        x = f_latents
        for name, w in zip(self.syn.layer_names[f_layer:], ws[1+f_layer:]):

            x = getattr(self.syn, name)(x, w, **layer_kwargs)
        # print(i, name, x.shape)
        # Ensure correct shape and dtype.
        if self.syn.output_scale != 1:
            x = x * self.syn.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.syn.img_channels, self.syn.img_resolution, self.syn.img_resolution])
        x = x.to(torch.float32)
        return x

    def forward_to_f(self, ws, f_layer, **layer_kwargs):
        misc.assert_shape(ws, [None, self.syn.num_ws, self.syn.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)


        # Execute layers.
        x = self.syn.input(ws[0])
        for name, w in zip(self.syn.layer_names[:f_layer], ws[1:1+f_layer]):
            x = getattr(self.syn, name)(x, w, **layer_kwargs)

        # print(i, name, x.shape)
        return x