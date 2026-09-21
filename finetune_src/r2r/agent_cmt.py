import json
import os
import sys
import numpy as np
import random
import re
import hashlib
#from random import *
import math
import time
from collections import defaultdict
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress

from models.model_HAMT import VLNBertCMT, Critic

from .eval_utils import cal_dtw

from .agent_base import BaseAgent

from transformers import get_linear_schedule_with_warmup

from utils.prompt_manager import PromptManager


class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    # for k, v in env_actions.items():
    #     env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank
        self.ranking_fp = None
        self._ranking_seen_states = set()
        if getattr(self.args, 'ranking_output', None) and self.rank in (-1, 0):
            os.makedirs(os.path.dirname(self.args.ranking_output), exist_ok=True)
            self.ranking_fp = open(self.args.ranking_output, 'w', buffering=1)
        self.mev_control_fp = None
        self._mev_entity_cache = {}
        self._mev_logprob_cache = {}
        if getattr(self.args, 'mev_control_output', None) and self.rank in (-1, 0):
            os.makedirs(os.path.dirname(self.args.mev_control_output), exist_ok=True)
            self.mev_control_fp = open(self.args.mev_control_output, 'w', buffering=1)

        # Models
        self._build_model()

        if not self.args.llm_predict:
            if self.args.world_size > 1:
                self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
                self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

            self.models = (self.vln_bert, self.critic)
            self.device = torch.device('cuda:%d'%self.rank) #TODO

            # Optimizers
            if self.args.optim == 'rms':
                optimizer = torch.optim.RMSprop
            elif self.args.optim == 'adam':
                optimizer = torch.optim.Adam
            elif self.args.optim == 'adamW':
                optimizer = torch.optim.AdamW
            elif self.args.optim == 'sgd':
                optimizer = torch.optim.SGD
            else:
                assert False
            if self.default_gpu:
                print('Optimizer: %s' % self.args.optim)

            self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)

            self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

            # Evaluations
            self.losses = []
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)


        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        if self.args.llm_predict:
            self.prompt_manager = PromptManager(self.args)

            local_accessory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LLaMA2-Accessory/accessory'))
            sys.path.append(local_accessory)
            from util.tensor_type import default_tensor_type
            from util.tensor_parallel import load_tensor_parallel_model_list
            from model.meta import MetaModel
            target_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[self.args.dtype]

            with default_tensor_type(dtype=target_dtype, device="cuda"):
                self.llm = MetaModel(self.args.llama_type, self.args.llama_config, self.args.tokenizer_path, max_seq_len=self.args.max_seq_len,with_visual=False)
            print(f"load pretrained from {self.args.pretrained_path}")
            load_result = load_tensor_parallel_model_list(self.llm, self.args.pretrained_path)
            print("load result: ", load_result)

            print("Model = %s" % str(self.llm))
            #self.llm.bfloat16().cuda()
            self.llm.cuda()
            if getattr(self.args, 'verifier_model_path', None):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                verifier_path = self.args.verifier_model_path
                print(f"load independent verifier from {verifier_path}")
                self.verifier_tokenizer = AutoTokenizer.from_pretrained(
                    verifier_path, local_files_only=True)
                self.verifier_llm = AutoModelForCausalLM.from_pretrained(
                    verifier_path, local_files_only=True, dtype=target_dtype
                ).cuda().eval()
        else:
            self.vln_bert = VLNBertCMT(self.args).cuda()
            self.critic = Critic(self.args).cuda()

    def get_direction(self, rel_heading, rel_elevation):
        if rel_elevation > 0:
            direction_text = 'go up to'
        elif rel_elevation < 0:
            direction_text = 'go down to'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    direction_text = 'turn left to'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    direction_text = 'go back to'
                else:
                    direction_text = 'turn right to'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    direction_text = 'turn right to'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    direction_text = 'go back to'
                else:
                    direction_text = 'turn left to'
            elif rel_heading == 0:
                direction_text = 'go forward to'
        return direction_text

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=np.float32))
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)

            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _candidate_variable(self, obs, previous_angle=None):

        if previous_angle is not None:
            batch_cand_index = []  #cand index(0, 12, 35...)
            batch_cand_action = []

        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            if previous_angle is not None:
                cand_index = []
                cand_action = []
                cand_vpids = []
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
                # cand_vpids.append(cc['viewpointId'])
                if previous_angle is not None:
                    direction = self.get_direction(cc['absolute_heading'] - previous_angle[i]['heading'],
                                                        cc['absolute_elevation'] - 0)
                    cand_index.append(cc['pointId'])

                    action_text = direction + ' to ' + f"<{cc['caption']}>"
                    cand_action.append(action_text)
            if previous_angle is not None:
                cand_action.append('stop')
                batch_cand_index.append(cand_index)
                batch_cand_action.append(cand_action)
                # batch_cand_vpids.append(cand_vpids)
            cand_nav_types[i, cand_lens[i]-1] = 2 # stop

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        if previous_angle is not None:
            return {
                'cand_img_feats':cand_img_feats,
                'cand_ang_feats':cand_ang_feats,
                'cand_nav_types':cand_nav_types,
                'cand_lens':cand_lens,
                'cand_action':batch_cand_action,
                'cand_index':batch_cand_index,
                # 'cand_vpids':batch_cand_vpids
            }
        else:
            return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _get_ig_probs(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        ig_probs = np.zeros((len(obs), max_len, self.args.ig_head), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                ig_probs[i, j] = cc['ig_probs']
                cand_nav_types[i, j] = 1
            ig_probs[i, cand_lens[i]-1] = cc['ig_probs']
            cand_nav_types[i, cand_lens[i]-1] = 2

        ig_probs = torch.from_numpy(ig_probs).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()

        return ig_probs, cand_nav_types

    def _get_ig_target(self, obs, ended):
        ig_target = np.zeros((len(obs), 196), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                ig_target[i,:] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:
                        ig_target[i,:] = candidate['ig_probs']
                        break
                else:
                    assert ob['teacher'] == ob['viewpoint']
                    ig_target[i,:] = self.args.ignoreid

        ig_target = torch.from_numpy(ig_target).cuda()

        return ig_target

    def _get_ig_probs_full(self, obs):
        ig_probs = np.zeros((len(obs), 196, self.args.ig_head), dtype=np.float32)
        # cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                if cc["viewpointId"] == ob['teacher']:
                    ig_probs[i] = cc["ig_probs"]
                    break
            else:
                assert ob['teacher'] == ob['viewpoint']         # All zeros probs

        ig_probs = torch.from_numpy(ig_probs).cuda()
        # cand_nav_types = torch.from_numpy(cand_nav_types).cuda()

        return ig_probs

    def _get_ig_probs_target(self, obs):
        ig_probs = np.zeros((len(obs), self.args.ig_head), dtype=np.float32)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                if cc["viewpointId"] == ob['teacher']:
                    ig_probs[i] = cc["ig_probs"]
                    break
            else:
                assert ob['teacher'] == ob['viewpoint']  # All zeros probs

        ig_probs = torch.from_numpy(ig_probs).cuda()

        return ig_probs

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                args = (name, 0, 0)
            else:                       # Adjust
                args = self.env_actions[name]
            try:
                self.env.env.sims[i].makeAction(*args)
            except TypeError:
                self.env.env.sims[i].makeAction([args[0]], [args[1]], [args[2]])

        def get_state(i):
            state = self.env.env.sims[i].getState()
            return state[0] if isinstance(state, (list, tuple)) else state

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while get_state(i).viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       get_state(i).navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = get_state(i)
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended)
                ml_loss += self.criterion(logit, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }
            _, last_h_ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj


    def rollout_ig(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        ig_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,  # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }

            if self.args.weighted_token:
                visual_inputs['position_id'] = torch.from_numpy(np.ones((batch_size, 196), dtype=np.float32)/196).to(ob_masks.get_device())

            t_outputs = self.vln_bert(**visual_inputs)
            logit_action = t_outputs[0]
            ig_probs = t_outputs[-1]            # batch_size, ig_head
            if self.args.weighted_token:
                dvae_target = self._get_ig_probs_target(obs)
                ig_probs = F.log_softmax(ig_probs, dim=-1)
            else:
                cand_probs, cand_masks = self._get_ig_probs(obs)

                cand_probs = torch.transpose(cand_probs, 1, 2)
                logit_dvae = torch.matmul(ig_probs, cand_probs).squeeze(1)
                logit_dvae.masked_fill_(cand_masks==0, -float('inf'))

            logit = logit_action
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            target = self._teacher_action(obs, ended)

            if train_ml is not None:
                # Supervised training
                ml_loss += self.criterion(logit, target)

            if self.args.weighted_token:
                ig_loss += F.kl_div(ig_probs, dvae_target, reduction='none').sum(dim=1).mean()
            else:
                ig_loss += self.criterion(logit_dvae, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i] - 1) or next_id == self.args.ignoreid or ended[
                    i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len - 1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t + 1, shortest_teacher=train_rl or self.feedback == 'argmax')

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }

            if self.args.weighted_token:
                visual_inputs['position_id'] = torch.from_numpy(np.ones((batch_size, 196), dtype=np.float32)/196).to(ob_masks.get_device())

            _, last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            ml_loss += self.args.train_ig * ig_loss
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())


        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj


    def rollout_llm(self, train_ml=None, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param reset:       Reset the environment

        :return:
        """

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        previous_angle = [{'heading': 0.,
                               'elevation': 0.} for ob in obs]


        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        #self.loss = 0

        for t in range(self.args.max_action_len):
            #print(t)
            if self.args.dataset == 'r2r':
                nav_targets = self._teacher_action(
                    obs=obs, ended=ended)

            if t == 0:
                for i in range(batch_size):
                    self.prompt_manager.history[i] = ''

            if self.args.ob_type == 'cand':
                cand_inputs = self._candidate_variable(obs = obs, previous_angle = previous_angle)

                nav_input = self.prompt_manager.get_prompt(mode = 'navigation', cand_inputs = cand_inputs, obs = obs, t = t)
                if 'nav_targets' in locals():
                    raw_gt_actions = [int(x) for x in nav_targets.detach().cpu().tolist()]
                    if self.args.stop_first:
                        self._ranking_gt_actions = [
                            0 if gt == cand_inputs['cand_lens'][i] - 1 else gt + 1
                            for i, gt in enumerate(raw_gt_actions)
                        ]
                    else:
                        self._ranking_gt_actions = raw_gt_actions
                else:
                    self._ranking_gt_actions = None
                self._ranking_obs = obs
                self._ranking_step = t
                self._ranking_instr_ids = [ob.get('instr_id') for ob in obs]
                if getattr(self.args, 'ranking_oracle_bank', False):
                    a_t_llm = self._score_oracle_ranking_bank(nav_input, obs, t)
                else:
                    num_samples = getattr(self.args, 'num_samples', 3)
                    all_nav_outputs = []
                    all_a_t_llm = []
                    for _ in range(num_samples):
                        nav_output = self.llm.generate(
                            nav_input["prompts"], images=None, max_gen_len=64,
                            temperature=max(self.args.temperature, 0.7),
                            top_p=getattr(self.args, 'top_p', 0.9)
                        )
                        all_nav_outputs.append(nav_output)
                        all_a_t_llm.append(self.prompt_manager.get_output(
                            nav_output=nav_output,
                            only_options_batch=nav_input["only_options"],
                            cand_inputs=cand_inputs, t=t
                        ))
                    a_t_llm = self._select_action_with_consistency_check(
                        all_a_t_llm, all_nav_outputs, nav_input, cand_inputs,
                        obs, t, batch_size
                    )
            else:
                assert False

            if self.feedback == 'teacher':
                a_t = nav_targets
            elif self.feedback == 'argmax':
                a_t = a_t_llm
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action

            cpu_a_t = a_t
            for i, next_id in enumerate(cpu_a_t):
                if self.args.stop_first:
                    if next_id == 0 or next_id == self.args.ignoreid or ended[i] or (t == self.args.max_action_len - 1):  # The last action is <end>
                        cpu_a_t[i] = -1  # Change the <end> and ignore action to -1
                    else:
                        cpu_a_t[i] = cpu_a_t[i]-1
                else:
                    if next_id == (cand_inputs['cand_lens'][i]-1) or next_id == self.args.ignoreid or ended[i] or (t == self.args.max_action_len - 1):    # The last action is <end>
                        cpu_a_t[i] = -1             # Change the <end> and ignore action to -1


            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)

            if self.feedback == 'teacher':
                obs = self.env._get_obs(t=t + 1)
            else:
                obs = self.env._get_obs(t=t + 1, shortest_teacher=True)

            previous_angle = [{'heading': ob['heading'],
                                   'elevation': ob['elevation']} for ob in obs]
            self.prompt_manager.make_history(a_t, nav_input, t)


            ended[:] = np.logical_or(ended, np.array([x == -1 for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            #self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.
            self.losses.append(self.loss / self.args.max_action_len)  # This argument is useless.

        return traj

    def _select_action_with_consistency_check(self, all_actions, all_nav_outputs, nav_input, cand_inputs, obs, t, batch_size):
        """
        检查多个候选动作的一致性，如果一致则直接返回，如果不一致则进行反向验证选择
        """
        final_actions = []
        
        # 初始化动作历史记录
        if not hasattr(self, 'action_history_batch'):
            self.action_history_batch = [[] for _ in range(batch_size)]
        
        for batch_idx in range(batch_size):
            # 收集这个batch中所有样本的动作
            actions_for_batch = [actions[batch_idx] for actions in all_actions]
            
            # 检查动作是否一致
            unique_actions = list(set(actions_for_batch))
            
            if (len(unique_actions) == 1 and self.ranking_fp is None
                    and self.mev_control_fp is None):
                # 所有动作都相同，直接使用
                selected_action = unique_actions[0]
                if getattr(self.args, 'verbose_consistency', False):
                    print(f"Step {t}, Batch {batch_idx}: All {len(actions_for_batch)} samples agree on action {selected_action}")
            else:
                # 动作不一致，需要进一步验证
                if getattr(self.args, 'verbose_consistency', False):
                    print(f"Step {t}, Batch {batch_idx}: Actions differ: {actions_for_batch}")
                
                # 调用反向验证函数
                selected_action = self._select_with_revision_protocol(
                    actions_for_batch,
                    [output[batch_idx] for output in all_nav_outputs],
                    nav_input,
                    cand_inputs,
                    obs[batch_idx],
                    t,
                    batch_idx
                )
            
            
            final_actions.append(selected_action)
        
        return final_actions

    def _select_with_revision_protocol(self, actions, outputs, nav_input, cand_inputs, obs, t, batch_idx):
        """Run the revision ablation with local LLaMA generation and verification."""
        mode = self.args.verification_mode
        if mode == 'direct':
            return actions[0]
        if mode == 'vote':
            return max(dict.fromkeys(actions), key=actions.count)

        instruction = obs.get('instruction', 'Navigation instruction')
        observation = ' '.join(nav_input['action_options'][batch_idx])
        history = self.prompt_manager.history[batch_idx] or 'None'
        candidates = []
        for action, output in zip(actions, outputs):
            if action not in [candidate[0] for candidate in candidates]:
                candidates.append((action, output))
        pairwise_mev = mode in ('mev_pairwise', 'dual_pairwise', 'mev_grounded', 'dual_grounded')
        mev_entities = (self._extract_mev_entities(
                            instruction, history=history,
                            observation=observation)
                        if mode in ('mev', 'dual', 'mev_pairwise', 'dual_pairwise', 'mev_grounded', 'dual_grounded') else [])

        scores = []
        score_records = []
        for action, output in candidates:
            tfv_raw = self._local_verification_score(
                f'Instruction: {instruction}\nObservation: {observation}\nHistory: {history}\n'
                f'Candidate reasoning: {output}\n'
                'Classify the candidate as TRUE if it is a plausible next step, otherwise FALSE. '
                'Output exactly one word: TRUE or FALSE. Do not output navigation reasoning.'
            ) if mode in ('tfv', 'dual', 'dual_pairwise', 'dual_grounded') else 0
            tfv_responses = list(getattr(self, '_last_verification_responses', []))
            tfv = (tfv_raw / self.args.verification_attempts
                   if mode in ('tfv', 'dual', 'dual_pairwise', 'dual_grounded')
                   else 0.0)
            if mode in ('mev', 'dual'):
                action_options = nav_input['action_options'][batch_idx]
                action_text = (action_options[action]
                               if 0 <= action < len(action_options)
                               else f'option {action}')
                mev, mev_responses = self._mev_recovery_score(
                    instruction, observation, history, action_text, mev_entities,
                    action_options=nav_input['action_options'][batch_idx],
                    action_index=action,
                        candidate_indices=[candidate[0] for candidate in candidates])
            elif pairwise_mev:
                action_options = nav_input['action_options'][batch_idx]
                action_text = (action_options[action]
                               if 0 <= action < len(action_options)
                               else f'option {action}')
                mev, mev_responses = self._mev_pairwise_score(
                    instruction, observation, history, action_text,
                    action, candidates, action_options, mev_entities,
                    grounded=mode in ('mev_grounded', 'dual_grounded'))
            else:
                mev, mev_responses = 0.0, []
            scores.append((action, tfv, mev))
            score_records.append({
                'action': int(action), 'tfv': float(tfv), 'mev': float(mev),
                'total': float(tfv + self.args.mev_weight * mev),
                'mev_weight': float(self.args.mev_weight),
                'tfv_responses': tfv_responses,
                'mev_responses': mev_responses, 'mev_entities': mev_entities,
                'mev_available': bool(mev_entities),
            })

        if self.mev_control_fp is not None:
            self._run_mev_language_controls(
                candidates=candidates,
                instruction=instruction,
                observation=observation,
                history=history,
                action_options=nav_input['action_options'][batch_idx],
                gt_action=(self._ranking_gt_actions[batch_idx]
                           if self._ranking_gt_actions is not None else None),
                instr_id=self._ranking_instr_ids[batch_idx],
                viewpoint=self._ranking_obs[batch_idx].get('viewpoint'),
                step=t,
            )

        if mode == 'mev_control':
            return actions[0]

        if self.ranking_fp is not None:
            gt = None
            if getattr(self, '_ranking_gt_actions', None) is not None:
                gt = self._ranking_gt_actions[batch_idx]
            rec = {
                'instr_id': self._ranking_instr_ids[batch_idx],
                'step': int(getattr(self, '_ranking_step', -1)),
                'viewpoint': self._ranking_obs[batch_idx].get('viewpoint'),
                'gt_action': int(gt) if gt is not None else None,
                'mode': mode,
                'verifier_type': ('independent_qwen' if getattr(self, 'verifier_llm', None)
                                  is not None else 'local_llama_generation'),
                'candidates': score_records,
            }
            self.ranking_fp.write(json.dumps(rec) + '\n')

        # Stable tie-break: TFV score, then original candidate order.
        selected = max(enumerate(scores), key=lambda item: (
            item[1][1] + self.args.mev_weight * item[1][2],
            item[1][1], -item[0]
        ))[1][0]
        if self.args.verbose_consistency:
            print(f'DV-VLN step={t} mode={mode} candidates={scores} selected={selected}')
        return selected

    def _local_verification_score(self, prompt):
        score = 0
        responses = []
        if getattr(self, 'verifier_llm', None) is not None:
            messages = [{'role': 'user', 'content': prompt}]
            text = self.verifier_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.verifier_tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = self.verifier_llm.generate(
                    **inputs, max_new_tokens=10, do_sample=True,
                    temperature=self.args.verification_temperature,
                    top_p=1.0, top_k=0,
                    num_return_sequences=self.args.verification_attempts,
                    pad_token_id=self.verifier_tokenizer.eos_token_id)
            answers = [self.verifier_tokenizer.decode(
                sequence[inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip() for sequence in generated]
        else:
            attempts = self.args.verification_attempts
            answers = self.llm.generate(
                [prompt] * attempts, images=None, max_gen_len=8,
                temperature=self.args.verification_temperature, top_p=1.0,
            )

        for answer in answers:
            if getattr(self, 'verifier_llm', None) is None:
                # The local same-model diagnostic requires a binary answer at
                # the beginning of the response; navigation text containing
                # the word "true" must not be counted as a positive judgment.
                match = re.match(r'\s*(true|false)\b', answer, re.IGNORECASE)
                parsed = bool(match and match.group(1).lower() == 'true')
                parseable = bool(match)
            else:
                parsed = self._parse_verification_response(answer)
                parseable = bool(re.fullmatch(
                    r'\s*(true|false)[.!]?\s*', answer, re.IGNORECASE))
            responses.append({
                'text': answer,
                'label': ('TRUE' if parsed else 'FALSE'),
                'parseable': parseable,
            })
            score += int(parsed)
        self._last_verification_responses = responses
        return score

    @staticmethod
    def _original_tfv_prompt(instruction, observation, history, output):
        return f"""你是一个顶尖的导航推理专家。请根据以下导航智能体的执行记录，判断智能体当前选择的下一步动作是否是正确的。

    输入：

    Instruction：{instruction}

    Observation：{observation}

    History：{history}

    输出：{output}

    基于以上这些分析，判断智能体当前选择的下一步动作是否是正确的。只需回答True或False。"""

    def _score_oracle_ranking_bank(self, nav_input, obs, step):
        """Score a deterministic oracle-plus-negatives bank and follow the oracle path."""
        selected_actions = []
        for batch_idx, ob in enumerate(obs):
            options = nav_input['action_options'][batch_idx]
            gt_action = self._ranking_gt_actions[batch_idx]
            state_key = (ob['instr_id'], int(step), ob['viewpoint'])
            if state_key in self._ranking_seen_states:
                selected_actions.append(gt_action)
                continue
            self._ranking_seen_states.add(state_key)
            scan = ob['scan']
            current = ob['viewpoint']
            goal = ob['gt_path'][-1]
            current_distance = self.env.shortest_distances[scan][current][goal]

            positive_actions = []
            if current_distance < 3.0:
                positive_actions.append(0 if self.args.stop_first else len(options) - 1)
            for candidate_idx, candidate in enumerate(ob['candidate']):
                action_idx = candidate_idx + 1 if self.args.stop_first else candidate_idx
                next_viewpoint = candidate['viewpointId']
                edge = self.env.shortest_distances[scan][current][next_viewpoint]
                remaining = self.env.shortest_distances[scan][next_viewpoint][goal]
                if abs(edge + remaining - current_distance) < 1e-5:
                    positive_actions.append(action_idx)
            positive_actions = sorted(set(positive_actions + [gt_action]))

            negatives = [idx for idx in range(len(options)) if idx not in positive_actions]
            seed_text = f"0:{ob['instr_id']}:{step}"
            seed_value = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
            rng = random.Random(seed_value)
            rng.shuffle(negatives)
            bank_actions = [gt_action] + negatives[:max(0, self.args.num_samples - 1)]
            rng.shuffle(bank_actions)

            observation = '[' + ', '.join(options) + ']'
            history = self.prompt_manager.history[batch_idx] or 'None'
            candidates = []
            for action in bank_actions:
                option_letter = chr(65 + action)
                candidate_output = (
                    f'Filtered observation: {options[action]}. Action: {option_letter}.'
                )
                prompt = self._original_tfv_prompt(
                    ob['instruction'], observation, history, candidate_output)
                score = self._local_verification_score(prompt)
                responses = list(self._last_verification_responses)
                candidates.append({
                    'action': int(action),
                    'action_text': options[action],
                    'is_positive': action in positive_actions,
                    'tfv': float(score),
                    'tfv_responses': responses,
                })

            record = {
                'instr_id': ob['instr_id'],
                'scan': scan,
                'step': int(step),
                'viewpoint': current,
                'goal_viewpoint': goal,
                'gt_action': int(gt_action),
                'positive_actions': positive_actions,
                'mode': 'tfv_oracle_bank',
                'verifier_type': (
                    'independent_qwen_reconstructed_original_tfv'
                    if getattr(self, 'verifier_llm', None) is not None
                    else 'same_model_dvvln_llama2_tfv'
                ),
                'candidates': candidates,
            }
            self.ranking_fp.write(json.dumps(record, ensure_ascii=False) + '\n')
            selected_actions.append(gt_action)
        return selected_actions

    @staticmethod
    def _normalize_mev_text(text):
        text = text.strip().split('\n', 1)[0]
        text = re.sub(r'^(answer|entity)\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-z0-9\s-]', ' ', text.lower())
        return ' '.join(text.split())

    def _extract_mev_entities(self, instruction, history=None, observation=None):
        """Extract the next unresolved landmark for the current navigation state."""
        cache_key = (instruction, history or '')
        if cache_key in self._mev_entity_cache:
            return self._mev_entity_cache[cache_key]
        prompt = (
            'Identify the next unresolved physical landmark in a navigation instruction. '
            'Use the completed action history to skip landmarks already passed. '
            'Copy an exact phrase from the instruction and output only that phrase.\n'
            f'Instruction: {instruction}\n'
            f'Completed history: {history or "None"}\n'
            f'Current observation: {observation or "None"}\n'
            'Next unresolved landmark:'
        )
        if getattr(self, 'verifier_llm', None) is not None:
            messages = [{'role': 'user', 'content': prompt}]
            text = self.verifier_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.verifier_tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = self.verifier_llm.generate(
                    **inputs, max_new_tokens=12, do_sample=False,
                    pad_token_id=self.verifier_tokenizer.eos_token_id)
            answer = self.verifier_tokenizer.decode(
                generated[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True).strip()
        else:
            answer = self.llm.generate(
                [prompt], images=None, max_gen_len=10, temperature=0, top_p=1.0
            )[0]
        proposed = self._normalize_mev_text(answer)
        candidates = []
        if (proposed and proposed != 'none'
                and re.search(rf'\b{re.escape(proposed)}\b', instruction,
                              flags=re.IGNORECASE)):
            candidates.append(proposed)

        # Restricted fallback for malformed extractor outputs. Unlike the earlier
        # unrestricted content-word heuristic, this list contains physical VLN entities.
        entity_lexicon = {
            'archway', 'balcony', 'bathroom', 'bed', 'bedroom', 'bench',
            'cabinet', 'chair', 'chandelier', 'closet', 'column', 'corridor',
            'couch', 'counter', 'desk', 'dining room', 'door', 'doorway',
            'entrance', 'entryway', 'fireplace', 'foyer', 'hall', 'hallway',
            'island', 'kitchen', 'landing', 'mirror', 'office', 'painting',
            'piano', 'railing', 'room', 'sofa', 'staircase', 'stairs', 'table',
            'television', 'window',
        }
        if not candidates:
            present = [entity for entity in entity_lexicon
                       if re.search(rf'\b{re.escape(entity)}\b', instruction,
                                    flags=re.IGNORECASE)]
            present.sort(key=lambda item: (-len(item), instruction.lower().find(item)))
            candidates.extend(present[:1])

        result = candidates[:max(1, self.args.mev_max_entities)]
        self._mev_entity_cache[cache_key] = result
        return result

    def _extract_mev_entities_legacy(self, instruction):
        """Deprecated unrestricted fallback retained for reference only."""
        stop_words = {
            'about', 'after', 'again', 'along', 'around', 'before', 'behind',
            'continue', 'down', 'enter', 'forward', 'from', 'front', 'head',
            'into', 'left', 'next', 'past', 'right', 'stop', 'straight',
            'then', 'through', 'toward', 'towards', 'turn', 'until', 'walk',
            'with', 'your', 'the', 'and', 'that', 'this', 'there', 'where',
            'you', 'are', 'for', 'near', 'onto', 'out', 'over', 'take', 'veer',
        }
        words = re.findall(r"[A-Za-z][A-Za-z-]+", instruction)
        unique = []
        for word in words:
            norm = word.lower()
            if len(norm) <= 2 or norm in stop_words or norm in unique:
                continue
            unique.append(norm)
        unique.sort(key=lambda item: (-len(item), words.index(next(
            word for word in words if word.lower() == item))))
        return unique[:max(1, self.args.mev_max_entities)]

    @staticmethod
    def _mask_mev_entity(instruction, entity):
        return re.sub(rf'\b{re.escape(entity)}\b', '[MASK]', instruction,
                      flags=re.IGNORECASE)

    @staticmethod
    def _make_mev_recovery_prompt(masked_instruction, observation, history,
                                  action_text=None):
        fields = [
            'Recover the exact entity replaced by [MASK].',
            'Example 1:',
            'Masked instruction: Walk past the [MASK] and enter the kitchen.',
            'Observation: A. stop B. go forward to a sofa C. turn right to a kitchen',
            'History: None',
            'Candidate action: B. go forward to a sofa',
            'Missing entity: sofa',
            'Example 2:',
            'Masked instruction: Leave the bedroom through the [MASK].',
            'Observation: A. stop B. go forward to an open door',
            'History: None',
            'Missing entity: door',
            'Query:',
            f'Masked instruction: {masked_instruction}',
            f'Observation: {observation}',
            f'History: {history}',
        ]
        if action_text is not None:
            fields.append(f'Candidate action: {action_text}')
        fields.append('Output only the missing entity.\nMissing entity:')
        return '\n'.join(fields)

    @staticmethod
    def _make_mev_scoring_prompt(masked_instruction, observation, history,
                                 action_text):
        return '\n'.join([
            'Recover the exact entity replaced by [MASK].',
            f'Masked instruction: {masked_instruction}',
            f'Observation: {observation}',
            f'History: {history}',
            f'Candidate action: {action_text}',
            'Output only the missing entity.\nMissing entity:',
        ])

    @staticmethod
    def _make_mev_pairwise_prompt(masked_instruction, target_entity,
                                  observation, history, candidate_action,
                                  counterfactual_action):
        """Compare which generated action better supports the current entity."""
        return '\n'.join([
            'You are evaluating two candidate navigation actions.',
            'Compare which action better supports reaching or observing the stated target entity now.',
            'Use the observation and history. The action text is the only variable that distinguishes the hypotheses.',
            'Example: if the first action is better, answer CANDIDATE; if the second is better, answer COUNTERFACTUAL.',
            'Output exactly CANDIDATE or COUNTERFACTUAL.',
            f'Masked instruction: {masked_instruction}',
            f'Current target entity: {target_entity}',
            f'Observation: {observation}',
            f'History: {history}',
            f'Candidate action: {candidate_action}',
            f'Counterfactual action: {counterfactual_action}',
            'Answer:',
        ])

    @staticmethod
    def _hide_mev_entity(text, entity):
        """Remove direct lexical evidence of the target from every comparison field."""
        return re.sub(rf'\b{re.escape(entity)}\b', '[MASKED-ENTITY]', text,
                      flags=re.IGNORECASE)

    def _mev_pairwise_trials(self, prompt):
        """Sample Qwen's action comparison and parse only the first decision token."""
        if getattr(self, 'verifier_llm', None) is not None:
            messages = [{'role': 'user', 'content': prompt}]
            text = self.verifier_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.verifier_tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = self.verifier_llm.generate(
                    **inputs, max_new_tokens=6, do_sample=True,
                    temperature=self.args.verification_temperature,
                    top_p=1.0, top_k=0,
                    num_return_sequences=self.args.verification_attempts,
                    pad_token_id=self.verifier_tokenizer.eos_token_id)
            answers = [self.verifier_tokenizer.decode(
                sequence[inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip() for sequence in generated]
        else:
            answers = self.llm.generate(
                [prompt] * self.args.verification_attempts, images=None,
                max_gen_len=6, temperature=max(self.args.temperature, 0.7),
                top_p=self.args.top_p)
        trials = []
        for answer in answers:
            # Prefer the required leading token, but recover labels from a
            # short explanatory response so formatting noise does not erase a
            # semantically usable verifier judgment.
            match = re.match(r'\s*(CANDIDATE|COUNTERFACTUAL)\b', answer,
                             flags=re.IGNORECASE)
            if match is None:
                match = re.search(r'\b(CANDIDATE|COUNTERFACTUAL)\b', answer,
                                  flags=re.IGNORECASE)
            label = match.group(1).upper() if match else 'INVALID'
            trials.append({
                'raw': answer,
                'label': label,
                'parseable': bool(match),
                'success': label == 'CANDIDATE',
            })
        return trials

    @staticmethod
    def _make_mev_likelihood_prompt(masked_instruction, observation, history,
                                    action_text):
        return '\n'.join([
            'Recover the exact physical landmark replaced by [MASK].',
            'Base the recovery on whether the candidate action is compatible with the current navigation state.',
            f'Masked instruction: {masked_instruction}',
            f'Observation: {observation}',
            f'History: {history}',
            f'Candidate action: {action_text}',
            'Missing entity:',
        ])

    def _qwen_target_logprob(self, prompt, target):
        """Mean teacher-forced log probability of target after a verifier prompt."""
        messages = [{'role': 'user', 'content': prompt}]
        chat = self.verifier_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.verifier_tokenizer(
            chat, return_tensors='pt', add_special_tokens=False
        ).input_ids.to('cuda')
        target_ids = self.verifier_tokenizer(
            ' ' + target, return_tensors='pt', add_special_tokens=False
        ).input_ids.to('cuda')
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        with torch.inference_mode():
            logits = self.verifier_llm(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            ).logits
        start = prompt_ids.shape[1] - 1
        target_logits = logits[:, start:start + target_ids.shape[1], :]
        token_logprobs = F.log_softmax(target_logits.float(), dim=-1).gather(
            -1, target_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_logprobs.mean().item())

    def _mev_recovery_trials(self, prompt, entity):
        target = self._normalize_mev_text(entity)
        if getattr(self, 'verifier_llm', None) is not None:
            messages = [{'role': 'user', 'content': prompt}]
            text = self.verifier_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.verifier_tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = self.verifier_llm.generate(
                    **inputs, max_new_tokens=12, do_sample=True,
                    temperature=self.args.verification_temperature,
                    top_p=1.0, top_k=0,
                    num_return_sequences=self.args.verification_attempts,
                    pad_token_id=self.verifier_tokenizer.eos_token_id)
            answers = [self.verifier_tokenizer.decode(
                sequence[inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip() for sequence in generated]
        else:
            prompts = [prompt] * self.args.verification_attempts
            max_batch_size = self.llm.llma.params.max_batch_size
            answers = []
            for start in range(0, len(prompts), max_batch_size):
                answers.extend(self.llm.generate(
                    prompts[start:start + max_batch_size], images=None, max_gen_len=12,
                    temperature=max(self.args.temperature, 0.7),
                    top_p=self.args.top_p,
                ))
        trials = []
        for answer in answers:
            prediction = self._normalize_mev_text(answer)
            trials.append({
                'raw': answer,
                'prediction': prediction,
                'success': prediction == target,
            })
        return trials

    def _mev_recovery_score(self, instruction, observation, history,
                            action_text, entities, action_options=None,
                            action_index=None, candidate_indices=None):
        """Return an action-conditioned masked-entity likelihood ratio.

        Language-only recoverability is removed with a no-action baseline. A
        positive margin means that conditioning on this candidate makes the
        target entity more likely than instruction context alone.
        """
        if not entities:
            return 0.0, []
        if getattr(self, 'verifier_llm', None) is None:
            return 0.0, []
        entity_records = []
        for entity in entities:
            masked_instruction = self._mask_mev_entity(instruction, entity)
            masked_observation = self._hide_mev_entity(observation, entity)
            masked_history = self._hide_mev_entity(history, entity)
            masked_candidate = self._hide_mev_entity(action_text, entity)
            candidate_prompt = self._make_mev_likelihood_prompt(
                masked_instruction, masked_observation, masked_history,
                masked_candidate)
            baseline_prompt = self._make_mev_likelihood_prompt(
                masked_instruction, masked_observation, masked_history,
                'None')
            candidate_logprob = self._qwen_target_logprob(
                candidate_prompt, entity)
            baseline_key = (masked_instruction, masked_observation,
                            masked_history, entity)
            if baseline_key not in self._mev_logprob_cache:
                self._mev_logprob_cache[baseline_key] = self._qwen_target_logprob(
                    baseline_prompt, entity)
            baseline_logprob = self._mev_logprob_cache[baseline_key]
            margin = candidate_logprob - baseline_logprob
            temperature = max(float(self.args.mev_logprob_temperature), 1e-6)
            normalized_score = 1.0 / (1.0 + math.exp(-margin / temperature))
            entity_records.append({
                'entity': entity,
                'masked_instruction': masked_instruction,
                'candidate_action': action_text,
                'masked_observation': masked_observation,
                'masked_history': masked_history,
                'candidate_logprob': candidate_logprob,
                'baseline_logprob': baseline_logprob,
                'logprob_margin': margin,
                'normalized_score': normalized_score,
            })
        score = float(np.mean([item['normalized_score']
                               for item in entity_records]))
        return score, entity_records

    def _mev_pairwise_score(self, instruction, observation, history,
                            action_text, action_index, candidates,
                            action_options, entities, grounded=False):
        """Score an action by contrastive entity support against peer actions.

        Unlike the likelihood-ratio variant, every comparison keeps the state
        and target fixed and changes only the candidate action.  This removes
        the common language-completion prior and makes the score explicitly
        relative to the alternatives proposed at this navigation step.
        """
        if not entities or getattr(self, 'verifier_llm', None) is None:
            return 0.0, []
        peers = [(idx, action) for idx, action in candidates if idx != action_index]
        if not peers:
            return 0.0, []
        records = []
        for entity in entities:
            masked_instruction = self._mask_mev_entity(instruction, entity)
            masked_observation = self._hide_mev_entity(observation, entity)
            masked_history = self._hide_mev_entity(history, entity)
            wins = 0
            trials = []
            for peer_index, _ in peers:
                peer_text = (action_options[peer_index]
                             if 0 <= peer_index < len(action_options)
                             else f'option {peer_index}')
                # Query both display orders for every pair. A fixed A/B
                # preference then cancels into a tie instead of becoming a
                # spurious action score; the parser maps each answer back to
                # the original candidate before aggregation.
                for swapped in (False, True):
                    first_action = peer_text if swapped else action_text
                    second_action = action_text if swapped else peer_text
                    prompt = self._make_mev_pairwise_prompt(
                        masked_instruction, entity, masked_observation,
                        masked_history, self._hide_mev_entity(first_action, entity),
                        self._hide_mev_entity(second_action, entity))
                    pair_trials = self._mev_pairwise_trials(prompt)
                    for item in pair_trials:
                        # `success` means the displayed first action won. If
                        # the display order was swapped, the original
                        # candidate wins when the displayed second action won.
                        item['candidate_won'] = (
                            not item['success'] if swapped else item['success'])
                        wins += int(item['candidate_won'])
                        item['swapped'] = swapped
                    trials.extend(pair_trials)
            total = len(trials)
            pairwise_score = wins / total if total else 0.0
            lexical = self._normalize_mev_text(entity) in self._normalize_mev_text(action_text)
            score = (0.5 * pairwise_score + 0.5 * float(lexical)
                     if grounded else pairwise_score)
            records.append({
                'entity': entity,
                'candidate_action': action_text,
                'wins': wins,
                'comparisons': total,
                'normalized_score': score,
                'pairwise_score': pairwise_score,
                'grounded_entity_match': bool(lexical),
                'trials': trials,
            })
        return float(np.mean([item['normalized_score'] for item in records])), records

    def _run_mev_language_controls(self, candidates, instruction, observation,
                                   history, action_options, gt_action, instr_id,
                                   viewpoint, step):
        entities = self._extract_mev_entities(instruction)
        if not entities:
            return
        for candidate_index, (action, _) in enumerate(candidates):
            action_text = (action_options[action] if 0 <= action < len(action_options)
                           else f'option {action}')
            alternatives = [
                option for idx, option in enumerate(action_options) if idx != action
            ]
            random_action = random.choice(alternatives) if alternatives else action_text
            condition_actions = {
                'full': action_text,
                'without_action': None,
                'random_action': random_action,
            }
            conditions = {}
            for condition, conditioned_action in condition_actions.items():
                entity_records = []
                for entity in entities:
                    masked = self._mask_mev_entity(instruction, entity)
                    prompt = self._make_mev_recovery_prompt(
                        masked, observation, history, conditioned_action
                    )
                    trials = self._mev_recovery_trials(prompt, entity)
                    successes = sum(int(trial['success']) for trial in trials)
                    entity_records.append({
                        'entity': entity,
                        'masked_instruction': masked,
                        'trials': trials,
                        'successes': successes,
                        'majority_correct': successes > len(trials) / 2,
                    })
                total_trials = sum(len(item['trials']) for item in entity_records)
                total_successes = sum(item['successes'] for item in entity_records)
                conditions[condition] = {
                    'conditioned_action': conditioned_action,
                    'normalized_score': (total_successes / total_trials
                                         if total_trials else 0.0),
                    'entity_records': entity_records,
                }
            self.mev_control_fp.write(json.dumps({
                'instr_id': instr_id,
                'step': int(step),
                'viewpoint': viewpoint,
                'candidate_index': candidate_index,
                'candidate_action': int(action),
                'candidate_action_text': action_text,
                'gt_action': int(gt_action) if gt_action is not None else None,
                'is_correct': bool(gt_action is not None and action == gt_action),
                'entities': entities,
                'conditions': conditions,
            }) + '\n')

    def _reverse_verification_selection(self, actions_for_batch, nav_outputs_for_batch, nav_input, cand_inputs, obs, t, batch_idx):
        """
        使用反向验证来选择最佳动作
        对每个候选动作分别验证3次，统计得分后选择最高分的动作
        """
        import json
        
        # 获取指令文本
        instruction = obs.get('instruction', '')
        if not instruction and 'instr_encoding' in obs:
            # 如果没有直接的指令文本，尝试从其他字段获取
            instruction = obs.get('instr_text', 'Navigation instruction')
        
        # 构建观察描述
        observation_parts = []
        if batch_idx < len(cand_inputs['cand_action']):
            cand_actions = cand_inputs['cand_action'][batch_idx]
            for i, action in enumerate(cand_actions):
                observation_parts.append(f"{chr(65+i)}.{action}")
        observation = "[" + " ".join(observation_parts) + "]"
        
        # 构建历史记录
        history = ""
        if hasattr(self, 'action_history_batch') and batch_idx < len(self.action_history_batch):
            for step_idx, action in enumerate(self.action_history_batch[batch_idx]):
                history += f"Step {step_idx + 1}. {action} "
        
        # 收集每个候选答案的验证结果
        candidate_scores = []
        
        for action_idx, (action, nav_output) in enumerate(zip(actions_for_batch, nav_outputs_for_batch)):
            # 对每个候选答案验证3次
            verification_count = 3
            true_count = 0
            
            # 构建验证prompt
            prompt = f"""你是一个顶尖的导航推理专家。请根据以下导航智能体的执行记录，判断智能体当前选择的下一步动作是否是正确的。

    输入：

    Instruction：{instruction}

    Observation：{observation}

    History：{history}

    输出：{nav_output}

    基于以上这些分析，判断智能体当前选择的下一步动作是否是正确的。只需回答True或False。"""

            # 对当前候选答案验证3次
            for verify_round in range(verification_count):
                try:
                    # 调用DeepSeek API
                    response = self._call_deepseek_api(prompt)
                    
                    # 解析响应
                    is_correct = self._parse_verification_response(response)
                    
                    if is_correct:
                        true_count += 1
                    
                    if getattr(self.args, 'verbose_consistency', False):
                        print(f"候选答案 {action_idx}, 第 {verify_round + 1} 次验证: {is_correct}")
                        
                except Exception as e:
                    if getattr(self.args, 'verbose_consistency', False):
                        print(f"验证候选答案 {action_idx} 第 {verify_round + 1} 次时出错: {e}")
                    # API调用失败时不增加分数
                    continue
            
            # 记录该候选答案的总分
            candidate_scores.append((action_idx, action, true_count))
            
            if getattr(self.args, 'verbose_consistency', False):
                print(f"候选答案 {action_idx} (action={action}): 总分 {true_count}/{verification_count}")
        
        # 选择得分最高的候选答案
        if not candidate_scores:
            # 如果所有验证都失败了，返回第一个动作作为默认值
            if getattr(self.args, 'verbose_consistency', False):
                print("所有验证都失败，返回第一个动作作为默认值")
            return actions_for_batch[0]
        
        # 找到最高分
        max_score = max(candidate_scores, key=lambda x: x[2])[2]
        
        # 获取所有最高分的候选答案
        best_candidates = [item for item in candidate_scores if item[2] == max_score]
        
        # 如果有多个最高分相同的候选答案，选择第一个
        selected_action = best_candidates[0][1]
        
        if getattr(self.args, 'verbose_consistency', False):
            if len(best_candidates) > 1:
                print(f"有 {len(best_candidates)} 个候选答案得分相同 (分数: {max_score})，选择第一个: {selected_action}")
            else:
                print(f"选择得分最高的候选答案: {selected_action} (分数: {max_score}/3)")
        
        return selected_action

    def _call_deepseek_api(self, prompt):
        """
        调用DeepSeek API进行验证
        """
        api_key = getattr(self.args, 'deepseek_api_key', None)
        if not api_key:
            raise ValueError("在 args.deepseek_api_key 中未找到 DeepSeek API 密钥")
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": getattr(self.args, 'deepseek_model', "deepseek-chat"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 10,  # 只需要True/False，所以很短
            "temperature": getattr(self.args, 'verification_temperature', 0.3),  # 稍微提高温度增加验证的多样性
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"DeepSeek API 错误: {response.status_code}, {response.text}")
    
    def _parse_verification_response(self, response):
        """
        解析验证响应，返回True或False
        """
        response = response.lower().strip()
        
        # Preserve the original repository parser for reconstruction fidelity.
        if 'true' in response:
            return True
        elif 'false' in response:
            return False

        positive_indicators = ['yes', 'correct', '正确', '是', '对', '正确的', '对的', '好的', '可以']
        negative_indicators = ['no', 'incorrect', 'wrong', '错误', '否', '不对', '错的', '不正确', '不好', '不可以']
        for indicator in positive_indicators:
            if indicator in response:
                return True
        for indicator in negative_indicators:
            if indicator in response:
                return False
        
        # 如果无法解析，默认返回False
        if getattr(self.args, 'verbose_consistency', False):
            print(f"警告: 无法解析验证响应: {response}")
        
        return False
    
    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if self.args.llm_predict:
            self.llm.eval()
        else:
            if use_dropout:
                self.vln_bert.train()
                self.critic.train()
            else:
                self.vln_bert.eval()
                self.critic.eval()
        if self.args.use_ig:
            super().test(iters=iters, rollout_function=self.rollout_ig)
        elif self.args.llm_predict:
            super().test(iters=iters, rollout_function=self.rollout_llm)
        else:
            super().test(iters=iters)

    def test_ig(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters, rollout_function=self.rollout_ig)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []

        if self.args.use_ig:
            rollout_function = self.rollout_ig
        else:
            rollout_function = self.rollout
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                rollout_function(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    rollout_function(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                rollout_function(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']

            new_dict = {}
            for k,v in state_dict.items():
                if not k.startswith('vln_bert.apwig_head') and not k.startswith('vln_bert.mapwig_head'):
                    new_dict[k] = v
            state_dict = new_dict

            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                for k in load_keys:
                    if k not in model_keys:
                        print("missing in model keys", k)
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
