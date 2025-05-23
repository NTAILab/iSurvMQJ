import numpy as np

import torch

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score

from scipy.integrate import trapezoid
from sklearn.model_selection import train_test_split

from .alpha_net import AlphaNet

class iSurvMQ:

    """
    iSurvMQ is an interval-based survival analysis model designed to handle uncertainty
    in event times by discretizing time into intervals and applying soft attention-based learning.

    This model supports various interval grid strategies, attention mechanisms, 
    regularization options, and stochastic training configurations.

    Args:
        method (str): If 'mean', then iSurvM is used.
                      If 'quantile', then iSurvQ is used.
        lr (float): Learning rate for the optimizer.
        num_epoch (int): Number of training epochs.
        size (float, optional): Proportion of the dataset used as queries; the remaining part 
                                (1 - size) is used as keys during attention computation.
        random_state (int): Seed for random number generation.
        lr_scheduler (bool): Whether to use a learning rate scheduler.
        reg_alpha (float): Regularization coefficient for attention weights.
        reg_pi (float): Regularization coefficient for probabilities.
        dropout_rate (float): Dropout rate applied during training.
        beta1 (float): Beta1 parameter for the Adam optimizer.
        beta2 (float): Beta2 parameter for the Adam optimizer.
        entropy_reg (float): Coefficient for entropy-based regularization.
        batch_rate (float): Fraction of the dataset to use in each mini-batch.
        mask_rate (float): Probability of masking attention weights during training.
        use_unique (bool): If True, only samples with unique event times are used during training.
        k (int, optional): Defines the width of intervals used for uncensored examples.
                           Specifically, 2 * k + 1 adjacent intervals are used around the event time.
                           If not set, k is randomly sampled per example in the range [3, 10].
        embed_dim (int): Dimensionality of the attention embedding space.
        num_generation (int): Specifies the number of probability generations (matrix M).
        top (float): Proportion of the generated samples with the highest (worst) loss values to be used in loss aggregation.
                     It used only for iSurvQ.

    """

    def __init__(
            self,
            lr=1e-1,
            num_epoch=100,
            num_generation=100,
            size=None,
            top=0.1,
            method='mean', # 'mean', 'quantile'
            random_state=42,
            lr_scheduler=True,
            reg_alpha = 0.0,
            reg_pi = 0.0,
            num_epoch_M_pi_B = 0,
            dropout_rate = 0.5,
            beta1 = 0.9,
            beta2 = 0.999,
            entropy_reg = 0.1,
            batch_rate = 0.1,
            mask_rate = 0.5,
            use_unique = False,
            k = None,
            embed_dim=64,
            ) -> None:
        
        # Model parameters
        self.method = method
        self.size = size
        self.num_epoch = num_epoch
        self.num_epoch_M_pi_B = num_epoch_M_pi_B
        self.num_generation = num_generation
        self.top = top
        self.random_state = random_state
        self.entropy_reg = entropy_reg
        self.batch_rate = batch_rate
        self.mask_rate = mask_rate
        self.use_unique = use_unique
        self.k = k
        self.embed_dim = embed_dim

        # Optimization parameters
        self.lr = lr
        self.reg_alpha = reg_alpha
        self.reg_pi = reg_pi
        self.dropout_rate = dropout_rate
        self.betas = (beta1, beta2)
        self.lr_scheduler = lr_scheduler

        # Internal state (initialized during training)
        self._M_pi_B_param = None
        self._alpha_net = None

    def _prepare_data(self, y):
        delta_str, time_str = y.dtype.names

        delta = y[delta_str]
        time = y[time_str]

        time_ind = np.searchsorted(self._interval_bounds, time)

        return delta, time, time_ind

    def _get_interval_bounds(self, time):
        # time = np.concatenate(([0], time))

        self._interval_bounds = np.unique(time)
        self.num_intervals = len(self._interval_bounds)
        
        return self._interval_bounds

    def _generate_probabilities(self, delta, time_ind, alpha=0.5, random=42):
        rng = np.random.default_rng(random)

        N = len(delta)
        pi = np.zeros((N, self.num_intervals + 1))

        # (delta = 1)
        uncensored_mask = delta == 1
        pi[uncensored_mask, time_ind[uncensored_mask]] = 1

        # (delta = 0)
        censored_mask = delta == 0
        start_indices = time_ind[censored_mask]
        num_censored = censored_mask.sum()

        dirichlet_probs = np.zeros((num_censored, self.num_intervals + 1))

        for i, start_idx in enumerate(start_indices):
            remaining_intervals = self.num_intervals + 1 - start_idx
            dirichlet_probs[i, start_idx:] = rng.dirichlet([alpha] * remaining_intervals)

        pi[censored_mask] = dirichlet_probs

        return pi

    def _create_optimizer(self, alpha_net):
        param_groups = []

        param_groups.append({"params": alpha_net.parameters(), "weight_decay": self.reg_alpha})

        return torch.optim.Adam(param_groups, betas=self.betas, lr=self.lr)
    

    def _create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5, last_epoch=-1)
    
    def _count_loss_mean_quantile(self, probs, delta, time_ind, k_per_sample=None, test=False):
        M, N_k, T = probs.shape

        if test:
            probs_selected = probs[:, torch.arange(N_k), time_ind]
        else:
            if k_per_sample is None:
                mask_selected = (torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) - self.k)) & \
                                (torch.arange(T).unsqueeze(0).to(probs.device) <= (time_ind.unsqueeze(1) + self.k))
            else:
                mask_selected = (torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) - k_per_sample.unsqueeze(1))) & \
                                (torch.arange(T).unsqueeze(0).to(probs.device) <= (time_ind.unsqueeze(1) + k_per_sample.unsqueeze(1)))
                
                mask_selected = mask_selected.unsqueeze(0)

            probs_selected = torch.sum(probs * mask_selected, dim=2)

        mask = torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) + 1)
        mask = mask.unsqueeze(0)
        probs_tail = torch.sum(probs * mask, dim=2)

        log_probs_selected = torch.log(probs_selected + 1e-10)
        log_probs_tail = torch.log(probs_tail + 1e-10)

        loss_per_i = -delta.unsqueeze(0) * log_probs_selected - (1 - delta).unsqueeze(0) * log_probs_tail
        
        return loss_per_i
    
    def _count_loss_pi_learning(self, probs, delta, time_ind, M_pi, k_per_sample=None, test=False):
        N_k, T = probs.shape
        
        if test:
            probs_selected = probs[torch.arange(N_k), time_ind]
        else:
            if k_per_sample is None:
                mask_selected = (torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) - self.k)) & \
                            (torch.arange(T).unsqueeze(0).to(probs.device) <= (time_ind.unsqueeze(1) + self.k))
            else:
                mask_selected = (torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) - k_per_sample.unsqueeze(1))) & \
                                (torch.arange(T).unsqueeze(0).to(probs.device) <= (time_ind.unsqueeze(1) + k_per_sample.unsqueeze(1)))

            probs_selected = torch.sum(probs * mask_selected, dim=1)

        mask_tail = torch.arange(T).unsqueeze(0).to(probs.device) >= (time_ind.unsqueeze(1) + 1)
        probs_tail = torch.sum(probs * mask_tail, dim=1)

        log_probs_selected = torch.log(torch.clamp_min(probs_selected, 1e-20))
        log_probs_tail = torch.log(torch.clamp_min(probs_tail, 1e-20))

        loss_per_i = -delta * log_probs_selected - (1 - delta) * log_probs_tail 

        if not test:
            entropy = -torch.sum(M_pi * torch.log(torch.clamp_min(M_pi, 1e-20)), dim=1)  
            loss_per_i = loss_per_i - self.entropy_reg * entropy
        
        return loss_per_i

    
    def _count_c_index(self, probs, delta, time):

        cumulative_proba = np.cumsum(probs, axis=1)

        survival_function = 1 - cumulative_proba

        integrated_сum_proba = np.array([trapezoid(survival_function[i], self._interval_bounds) for i in range(survival_function.shape[0])])

        c_index = concordance_index_censored(delta.astype(bool), time, -integrated_сum_proba)

        return c_index[0]
    
    def count_ibs(self, X_test, y_train, y_test):

        time_str = y_train.dtype.names[1]

        min_time, max_time = y_test[time_str].min(), y_test[time_str].max()
        tolerance = 0.1 * (max_time - min_time)
        times = np.linspace(min_time + tolerance, max_time - tolerance, 100)

        survs = np.array([S(times) for S in self.predict_survival_function(X_test)])

        ibs = integrated_brier_score(y_train, y_test, survs, times)

        return ibs

    def fit(self, X, y, X_test=None, y_test=None):
        torch.manual_seed(self.random_state)

        delta_str, time_str = y.dtype.names

        if self.use_unique:
            _, self._train_indices = np.unique(y[time_str], return_index=True)

            y = y[self._train_indices]
            X = X[self._train_indices]

        self._get_interval_bounds(y[time_str])

        if self.size is not None:
            self._X_B, X_K, y_B, y_K = train_test_split(X, y, test_size=self.size, random_state=42, stratify=y[delta_str])
        else:
            self._X_B, y_B = X, y
            X_K, y_K = X, y

        delta_B, time_B, time_ind_B = self._prepare_data(y_B)  # key
        delta_K, time_K, time_ind_K = self._prepare_data(y_K)  # query

        time_ind_K, delta_K = map(lambda x: torch.tensor(x, dtype=torch.int32, device="cpu"), [time_ind_K, delta_K])
        time_ind_B, delta_B = map(lambda x: torch.tensor(x, dtype=torch.int32, device="cpu"), [time_ind_B, delta_B])

        M, N_b, T = 1, len(time_B), self.num_intervals

        self._alpha_net = AlphaNet(self._X_B.shape[1], self.embed_dim, dropout_rate=self.dropout_rate).to("cpu")

        optimizer = self._create_optimizer(self._alpha_net)
        scheduler = self._create_scheduler(optimizer) if self.lr_scheduler else None

        X_B_tensor, X_K_tensor = map(lambda x: torch.tensor(x, dtype=torch.float32, device="cpu"), [self._X_B, X_K])

        if X_test is not None and y_test is not None:
            delta_test, time_test, time_ind_test = self._prepare_data(y_test)
            time_ind_test, delta_test = map(lambda x: torch.tensor(x, dtype=torch.int32, device="cpu"), [time_ind_test, delta_test])
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device="cpu")

        self._train_losses, self._test_losses = [], [] if X_test is not None and y_test is not None else None
        all_generations = []
        self._test_ibs = []
        self._test_c_index = []

        self.batch_size = int(N_b * self.batch_rate)

        for iteration in range(self.num_epoch):

            indices = torch.randperm(N_b, device="cpu")
            self._batch_indices = torch.split(indices, self.batch_size)

            for batch_num, batch_idx in enumerate(self._batch_indices):

                optimizer.zero_grad()

                attn_mask = torch.rand(time_ind_B.shape[0], time_ind_K.shape[0], device="cpu")
                attn_mask = attn_mask > self.mask_rate
                attn_mask.fill_diagonal_(0)
                alpha = self._alpha_net(X_B_tensor, X_K_tensor, attn_mask)

                print(self.num_generation)

                M_pi_B = np.array([self._generate_probabilities(delta_B, time_ind_B, random=(self.num_epoch * self.random_state * i)) for i in range(self.num_generation)]) # M * N_b * T
                M_pi_B = torch.tensor(M_pi_B, dtype=torch.float32, device="cpu")

                alpha = alpha.unsqueeze(0).repeat(self.num_generation, 1, 1)
                probs = torch.einsum('mkb,mbt->mkt', alpha, M_pi_B)

                if self.k is None:
                    k_per_sample = torch.randint(3, 10, (delta_K.shape[0],))
                    loss_per_i = self._count_loss_mean_quantile(probs, delta_K, time_ind_K, k_per_sample)
                else:
                    loss_per_i = self._count_loss_mean_quantile(probs, delta_K, time_ind_K)
         
                if self.method == 'mean':
                    loss_per_i_mean = torch.mean(loss_per_i, dim=0)
                    loss = torch.mean(loss_per_i_mean) 
                    current_M_pi_B = M_pi_B.clone()
                elif self.method == 'quantile':
                    loss_per_m = torch.mean(loss_per_i, dim=1)
                    k = int(len(loss_per_m) * self.top)
                    top_losses, top_indices = torch.topk(loss_per_m, k=k, largest=True)
                    loss = torch.mean(top_losses)
                    current_M_pi_B = M_pi_B[top_indices.cpu().numpy()].clone()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._alpha_net.parameters(), max_norm=100)

                optimizer.step()

                ############ Batch End ############

            if self.lr_scheduler:
                scheduler.step()

            all_generations.append(current_M_pi_B.cpu().numpy())

            self._train_losses.append(loss.item())

            ### Test Loss ###
            if X_test is not None and y_test is not None:

                self._alpha_net.eval()

                with torch.no_grad():
                    test_current_M_pi_B = np.mean(np.array(all_generations), axis=(0, 1))
                    test_current_M_pi_B_tensor = torch.tensor(test_current_M_pi_B, dtype=torch.float32, device="cpu")

                    test_alpha = self._alpha_net(X_B_tensor, X_test_tensor)

                    test_probs = torch.einsum('kb,bt->kt', test_alpha, test_current_M_pi_B_tensor)
                    test_probs = test_probs.unsqueeze(0)
                    test_loss_per_i = self._count_loss_mean_quantile(test_probs, delta_test, time_ind_test, test=True)

                    c_index = self._count_c_index(test_probs.squeeze(0)[:, :-1], delta_test.cpu().numpy(), time_test)

                    self._M_pi_B = test_current_M_pi_B
                    ibs = self._count_ibs(X, X_test, y, y_test)

                    test_loss = torch.mean(torch.mean(test_loss_per_i, dim=0))
                    self._test_losses.append(test_loss.item())

                    self._test_c_index.append(c_index)
                    self._test_ibs.append(ibs)

                print(f"Epoch {iteration + 1}/{self.num_epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}, Test c-index: {c_index}, Test ibs: {ibs}")
            else:
                print(f"Epoch {iteration + 1}/{self.num_epoch}, Loss: {loss.item()}")

        all_generations_numpy = np.array(all_generations)

        self._M_pi_B = np.mean(all_generations_numpy, axis=(0, 1))

        if self.num_epoch_M_pi_B:

            mean_initial = np.mean(all_generations_numpy, axis=(0, 1))
            mean_initial_tensor = torch.tensor(mean_initial, dtype=torch.float32, device="cpu").unsqueeze(0)

            self._M_pi_B_param = torch.nn.Parameter(torch.log(torch.maximum(mean_initial_tensor, torch.tensor(10e-20, dtype=torch.float32, device="cpu"))))

            optimizer_pi = torch.optim.Adam([self._M_pi_B_param], lr=self.lr, weight_decay=self.reg_pi, betas=self.betas)

            for iteration in range(self.num_epoch_M_pi_B):
                optimizer_pi.zero_grad()

                M_pi_B = torch.softmax(self._M_pi_B_param, dim=2).squeeze(0)

                M_pi_B_new = M_pi_B.clone()

                # # (delta = 1)
                uncensored_mask = delta_B == 1
                M_pi_B_new[uncensored_mask, :] = 0
                M_pi_B_new[uncensored_mask, time_ind_B[uncensored_mask]] = 1

                # (delta = 0)
                censored_mask = delta_B == 0
                start_indices = time_ind_B[censored_mask]
                ids = torch.where(censored_mask)[0]
                for i, start in enumerate(start_indices):
                    M_pi_B_new[ids[i], :start] = 0

                M_pi_B_new_norm = M_pi_B_new / M_pi_B_new.sum(dim=1, keepdim=True)

                with torch.no_grad():
                    alpha = self._alpha_net(X_B_tensor, X_K_tensor)

                probs = torch.einsum('kb,bt->kt', alpha, M_pi_B_new_norm) 

                ### Train Loss ###

                if self.k is None:
                    k_per_sample = torch.randint(3, 10, (delta_K.shape[0],))
                    loss_per_i = self._count_loss_pi_learning(probs, delta_K, time_ind_K, M_pi_B_new_norm, k_per_sample)
                else:
                    loss_per_i = self._count_loss_pi_learning(probs, delta_K, time_ind_K, M_pi_B_new_norm)

                loss_per_i_mean = torch.mean(loss_per_i, dim=0)
                loss = torch.mean(loss_per_i_mean)

                loss.backward()
                optimizer_pi.step()

                if self.lr_scheduler:
                    scheduler.step()

                self._train_losses.append(loss.item())

                ### Test Loss ###
                if X_test is not None and y_test is not None:

                    self._alpha_net.eval() 

                    with torch.no_grad():

                        test_current_M_pi_B_tensor = M_pi_B_new_norm.clone()

                        test_alpha = self._alpha_net(X_B_tensor, X_test_tensor)
                        
                        test_probs = torch.einsum('kb,bt->kt', test_alpha, test_current_M_pi_B_tensor)

                        test_loss_per_i = self._count_loss_pi_learning(test_probs, delta_test, time_ind_test, test_current_M_pi_B_tensor, test=True)
                        test_loss = torch.mean(torch.mean(test_loss_per_i, dim=0))

                        self._test_losses.append(test_loss.item())

                    c_index = self._count_c_index(test_probs.squeeze(0)[:, :-1], delta_test.cpu().numpy(), time_test)

                    self._M_pi_B = test_current_M_pi_B
                    ibs = self._count_ibs(X, X_test, y, y_test)

                    self._test_c_index.append(c_index)
                    self._test_ibs.append(ibs)

                    print(f"Epoch for learning probs {iteration + 1}/{self.num_epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}, Test c-index: {c_index}, Test ibs: {ibs}")
                else:
                    print(f"Epoch for learning probs {iteration + 1}/{self.num_epoch_M_pi_B}, Loss: {loss.item()}")

            self._M_pi_B = M_pi_B_new_norm.squeeze(0).detach().cpu().numpy()

        return self

    def predict(self, X_new):
        self._alpha_net.eval() 

        keys_tensor, X_new_tensor = map(lambda x: torch.tensor(x, dtype=torch.float32, device="cpu"), [self._X_B, X_new])

        with torch.no_grad():
            alpha = self._alpha_net(keys_tensor, X_new_tensor)
            M_pi_B = torch.tensor(self._M_pi_B, dtype=torch.float32, device="cpu")
            probs = torch.einsum("kb, bt->kt", alpha, M_pi_B[:, :-1])
            
        return probs.detach().cpu().numpy()

    def get_weights(self, X):
        self._alpha_net.eval() 

        keys_tensor, X_tensor = map(lambda x: torch.tensor(x, dtype=torch.float32, device="cpu"), [self._X_B, X])

        with torch.no_grad():
            alpha = self._alpha_net(keys_tensor, X_tensor)

        return alpha.detach().cpu().numpy()

    def score(self, X, y):
        delta_str, time_str = y.dtype.names

        delta = y[delta_str]
        time = y[time_str]
        
        predicted_proba = self.predict(X)

        c_index = self._count_c_index(predicted_proba, delta, time)

        return c_index
    
    def _step_function(self, times, survival_function):
        if isinstance(times, (int, float)):
            times = [times]

        survs = []

        for time in times:
            if time < 0:
                raise ValueError("Time can't have negative value")

            if time < self._interval_bounds[0]:
                survs.append(1)
            elif time >= self._interval_bounds[-1]:
                survs.append(survival_function[-1])
            else:
                for i, bound in enumerate(self._interval_bounds):
                    if time < bound:
                        survs.append(survival_function[i - 1])
                        break

        return survs
    
    def predict_survival_function(self, X):
        predicted_proba = self.predict(X)

        cumulative_proba = np.cumsum(predicted_proba, axis=1)
        cumulative_proba[cumulative_proba > 1.0] = 1.0

        survival_functions = 1 - cumulative_proba

        step_functions = np.array([
            lambda x, sf=sf: self._step_function(x, sf) for sf in survival_functions
        ])
        
        return step_functions

    def get_params(self, deep=True):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)