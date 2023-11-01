import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

class SAC():
    def __init__(
        self,
        model,
        n_actions,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.98,
        memory_size = 5000,
        batch_size = 64,
        tau = 0.01,
        alpha = 0.5,
        auto_entropy_tuning = True,
        criterion = nn.MSELoss()
    ):
        # initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self._build_net(model[0], model[1])
        self.init_memory()

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(self.device)
        self.critic_target.eval()

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0001)
    
    def save_load_model(self, op, path, epi):
        anet_path = path + f"sac_anet_{str(epi).zfill(4)}.pt"
        cnet_path = path + f"sac_cnet_{str(epi).zfill(4)}.pt"
        if op == "save":
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.critic_target.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.actor.load_state_dict(torch.load(anet_path, map_location=self.device))

    def choose_action(self, s, eval=False):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        s_ts = {
            "rp": torch.FloatTensor(np.expand_dims(s["rp"], 0)).to(self.device),
            "obs": transform(s["obs"]).unsqueeze(0).to(self.device)
        }
        
        if eval == False:
            action, _, _ = self.actor.sample(s_ts)
        else:
            _, _, action = self.actor.sample(s_ts)
        
        action = action.cpu().detach().numpy()[0]
        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":{}, "a":[], "r":[], "sn":{}, "end":[]}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            for key in s:
                if key in self.memory["s"]:
                    self.memory["s"][key].append(s[key])
                else:
                    self.memory["s"][key] = [s[key]]
            for key in sn:
                if key in self.memory["sn"]:
                    self.memory["sn"][key].append(sn[key])
                else:
                    self.memory["sn"][key] = [sn[key]]
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            for key in s:
                self.memory["s"][key][index] = s[key]
            for key in sn:
                self.memory["sn"][key][index] = sn[key]
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]
        s_batch = {key: [self.memory["s"][key][index] for index in sample_index] for key in self.memory["s"]}
        sn_batch = {key: [self.memory["sn"][key][index] for index in sample_index] for key in self.memory["sn"]}

        # Construct torch tensor
        s_ts, sn_ts = {}, {}
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        a_ts = torch.FloatTensor(np.array(a_batch)).to(self.device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(self.device).view(self.batch_size, 1)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(self.device).view(self.batch_size, 1)
        s_ts = {
            "rp": torch.FloatTensor(np.array(s_batch["rp"])).to(self.device),
            "obs": torch.stack([transform(img) for img in s_batch["obs"]], dim=0).to(self.device)
        }
        sn_ts = {
            "rp": torch.FloatTensor(np.array(sn_batch["rp"])).to(self.device),
            "obs": torch.stack([transform(img) for img in sn_batch["obs"]], dim=0).to(self.device)
        }

        # TD-target
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(sn_ts)
            q_next_target = self.critic_target(sn_ts, a_next) - self.alpha * logpi_next
            q_target = r_ts + end_ts * self.gamma * q_next_target
        
        # Critic loss
        q_eval = self.critic(s_ts, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        a_curr, logpi_curr, _ = self.actor.sample(s_ts)
        q_current = self.critic(s_ts, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        
        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())
        
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())