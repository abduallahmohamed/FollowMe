import torch
import torch.nn as nn
import torch.distributions as tdist


class SocialCellLocal(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=5,
                 temporal_input=10,
                 temporal_output=30):
        super(SocialCellLocal, self).__init__()

        #Spatial Section
        self.feat = nn.Conv1d(spatial_input,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        self.feat_act = nn.ReLU()
        self.highway_input = nn.Conv1d(spatial_input,
                                       spatial_output,
                                       1,
                                       padding=0)

        #Temporal Section
        self.highway = nn.Conv1d(temporal_input, temporal_output, 1, padding=0)
        self.tpcnn = nn.Conv1d(temporal_input,
                               temporal_output,
                               3,
                               padding=1,
                               padding_mode='zeros')

        #Conditional IMLE Section
        self.w = nn.Parameter(torch.randn(1), requires_grad=True)
        self.noise_cnn = nn.Conv2d(temporal_input * 2,
                                   temporal_input,
                                   3,
                                   padding=1)
        self.noise = tdist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), torch.Tensor([[1, 0], [0, 1]]))

    def forward(self, x):
        x_lead = x[:, :, 1:2, :2]
        noise = self.noise.sample((1, )).unsqueeze(0).unsqueeze(0).repeat(
            1, x_lead.shape[1], 1, 1).to(x.device).contiguous()
        condition = torch.cat((x_lead, noise), dim=1)
        condition = self.feat_act(self.noise_cnn(condition))

        v = x[..., :2] + self.w * condition
        #Spatial Section
        v = v.permute(0, 2, 3, 1).squeeze(0)  #= PED*batch,  [x,y], TIME,
        v_res = self.highway_input(v)
        v = self.feat_act(self.feat(v)) + v_res

        #Temporal Section
        v = v.permute(0, 2, 1)
        v_res = self.highway(v)
        v = self.tpcnn(v) + v_res

        #Final Output
        v = v.permute(1, 0, 2).unsqueeze(0)
        return v


class FusionAttentionCNN(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=10,
                 temporal_output=30,
                 full_dim=5):
        super(FusionAttentionCNN, self).__init__()

        self.act = nn.ReLU()
        self.act_limit = nn.Sigmoid()
        self.cnn_locaclass_embedding = nn.Conv2d(full_dim,
                                                 spatial_output,
                                                 3,
                                                 padding=1)
        self.cnn_fuse = nn.Conv2d(temporal_input + temporal_output,
                                  temporal_output,
                                  3,
                                  padding=1)
        # self.fusion_weight

    def forward(
        self, v, v_spatial
    ):  #batch,time,ped,[x,y,class] || batch,time,1,[x,y] || batch, time, P-1, [x,y]
        v = v.permute(0, 3, 1, 2)
        v_spatial = v_spatial.permute(0, 1, 3, 2)

        v_loc_class_emb = self.act(self.cnn_locaclass_embedding(v)).permute(
            0, 2, 1, 3)  #Get embedding with class info
        v_fused = self.act_limit(
            self.cnn_fuse(torch.cat((v_spatial, v_loc_class_emb),
                                    dim=1)))  #Fuse it with the spatial info
        v_fused = v_fused.unsqueeze(-2)[..., 1:]
        v_spatial_others = v_spatial.unsqueeze(-2)[..., 1:].transpose(-1, -2)
        v_contribution = torch.matmul(
            v_fused, v_spatial_others).squeeze(-1).squeeze(
                -1)  #Create a self attention with the spatial info

        v_final = v_spatial[
            ..., 0] + v_contribution  #Add contribution to the ego v traj
        return v_final


class FollowMeSTGCNN(nn.Module):
    def __init__(self, obs_time=10, pred_time=30, full_dim=5, f_in=2, f_out=2):
        super(FollowMeSTGCNN, self).__init__()

        self.local = SocialCellLocal(f_in, f_out, obs_time, pred_time)
        self.fusion_attn = FusionAttentionCNN(f_in, f_out, obs_time, pred_time,
                                              full_dim)

    def forward(self, x):
        x_loc = self.local(x)
        x_loc = self.fusion_attn(x, x_loc)

        return x_loc