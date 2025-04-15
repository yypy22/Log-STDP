class LogSTDP:
    def __init__(self, W0=0.25, tau_plus=17, tau_minus=34,
                 gamma=50.0, S=5, c_plus=1.0, c_minus=0.5, w_max=1.0, sigma=0.6, lr=0.1):
        self.W0 = W0
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.gamma = gamma
        self.S = S
        self.c_plus = c_plus
        self.c_minus = c_minus
        self.w_max = w_max
        self.sigma = sigma
        self.homeostatic_scale = 1.0
        self.lr = lr

    def weight_change(self, w, delta_t):
        noise = torch.randn_like(w) * self.sigma
        delta_w_plus = self.c_plus * torch.exp(-w / (self.W0 * self.gamma)) * (delta_t < 0).float()
        small_w_mask = (w <= self.W0).float()
        large_w_mask = (w > self.W0).float()
        delta_w_minus_small = self.c_minus * (w / self.W0) * small_w_mask * (delta_t > 0).float()
        delta_w_minus_large = self.c_minus * (1 + torch.log(1 + self.S * (w / self.W0 - 1)) / self.S) * large_w_mask * (delta_t > 0).float()
        delta_w_minus = delta_w_minus_small + delta_w_minus_large
        delta_w = self.lr*(1+noise) * (delta_w_plus*torch.exp(-np.abs(delta_t)/self.tau_plus) - (delta_w_minus)*torch.exp(-np.abs(delta_t)/self.tau_minus))
        return delta_w

    def update_weight(self, w, delta_t):
        delta_w = self.weight_change(w, delta_t)
        new_w = w + delta_w
        return new_w
