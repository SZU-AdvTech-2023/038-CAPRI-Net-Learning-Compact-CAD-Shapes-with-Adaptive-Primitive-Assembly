import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
	def __init__(self, ef_dim=32, p_dim=1024):
		super(Decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
		self.linear_3 = nn.Linear(self.ef_dim*32, self.p_dim*7, bias=True)
		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.xavier_normal_(self.linear_3.weight)
		nn.init.constant_(self.linear_3.bias, 0)

	def forward(self, inputs):
		l1 = self.linear_1(inputs)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)
		
		l3 = self.linear_3(l2)
		l3 = l3.view(-1, 7, self.p_dim)
		l3 = torch.cat([torch.abs(l3[:, :3, :]), l3[:, 3:, :]], 1)
		return l3