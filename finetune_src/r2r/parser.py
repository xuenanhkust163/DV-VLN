import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument(
        '--dataset', type=str, default='r2r',
        choices=['r2r', 'r4r', 'r2r_back', 'r2r_last', 'rxr']
    )
    parser.add_argument('--langs', nargs='+', default=None, choices=['en', 'hi', 'te'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")

    # General
    parser.add_argument('--iters', type=int, default=300000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=2000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    parser.add_argument('--ob_type', type=str, choices=['cand', 'pano'], default='pano')
    parser.add_argument('--vlnbert', type=str, default='cmt',
                        choices=['cmt', 'causal.cmt', 'mmt', 'cmt3cat', 'cmt3stack'])
    parser.add_argument('--cfg_name', type=str, default='bert-base-uncased',
                        help='Hugging Face name or local path for the VLN text backbone.')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--eval_splits', nargs='+', default=None,
                        help='Validation splits to evaluate; defaults to the legacy split set.')
    parser.add_argument('--eval_max_instrs', type=int, default=None,
                        help='Optional per-split instruction cap for diagnostic smoke tests.')

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')

    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument("--teacher_weight", type=float, default=1.)

    parser.add_argument("--features", type=str, default='places365')
    parser.add_argument("--features_aug", default=None)
    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_hist_embedding', action='store_true', default=False)
    parser.add_argument('--fix_obs_embedding', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_h_layers', type=int, default=0)
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--hist_enc_pano', action='store_true', default=False)
    parser.add_argument('--hist_pano_num_layers', type=int, default=2)
    # cmt
    parser.add_argument('--no_lang_ca', action='store_true', default=False)
    parser.add_argument('--act_pred_token', default='ob_txt', choices=['ob', 'ob_txt', 'ob_hist', 'ob_txt_hist'])

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_cand_backtrack', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument(
        '--teacher', type=str, default='final',
        help="How to get supervision. one of ``next`` and ``final`` "
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--views', type=int, default=36)

    # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total",
        type=str, help='batch or total'
    )

    parser.add_argument("--use_ig", action='store_true', default=False)
    parser.add_argument("--ig_head", type=int, default=8192)
    parser.add_argument("--ig_path", default=None)
    parser.add_argument("--weighted_token", action='store_true', default=False)
    parser.add_argument("--train_ig", default=1, type=float)
    parser.add_argument("--step_eval", action='store_true', default=False)

    # llm
    parser.add_argument("--llm_predict", action='store_true', default=False)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument("--stop_first", action='store_true', default=False)
    parser.add_argument('--temperature', type=float,default=0)
    parser.add_argument('--candidate_cap_dir', type=str, default=None,
                        help='Candidate-caption root; defaults to ROOT_DIR/R2R/captions.')
    #llama2
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default=[], type=str, nargs="*",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default=None, type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="The dtype used for model weights and inference.")
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    # RXR
    parser.add_argument('--only_en', action='store_true', default=False)

    parser.add_argument('--num_samples', type=int, default=4,
                    help='K: sampled navigation candidates per step')
    parser.add_argument('--top_p', type=float, default=0.9,
                    help='Nucleus-sampling probability for candidate generation')
    parser.add_argument('--verification_attempts', type=int, default=4,
                    help='P: repeated local verification samples per candidate')
    parser.add_argument('--verification_temperature', type=float, default=0.3,
                    help='Sampling temperature for the independent verifier.')
    parser.add_argument('--verification_mode', choices=['direct', 'vote', 'tfv', 'mev', 'dual', 'mev_pairwise', 'dual_pairwise', 'mev_grounded', 'dual_grounded', 'mev_control'], default='dual',
                    help='Revision ablation mode: no verifier, vote, TFV, MEV, or TFV+MEV.')
    parser.add_argument('--verbose_consistency', action='store_true', default=False,
                    help='Print candidate actions and selection decisions.')
    parser.add_argument('--verbose_verification', action='store_true', default=False,
                        help='是否打印反向验证的详细信息')
    parser.add_argument('--ranking_output', type=str, default=None,
                        help='Optional JSONL path for verifier ranking diagnostics.')
    parser.add_argument('--ranking_oracle_bank', action='store_true', default=False,
                        help='Score one oracle action and up to three legal negatives per state.')
    parser.add_argument('--verifier_model_path', type=str, default=None,
                        help='Optional Hugging Face causal LM used as an independent verifier.')
    parser.add_argument('--mev_control_output', type=str, default=None,
                        help='Optional JSONL path for Full/no-action/random-action MEV controls.')
    parser.add_argument('--mev_max_entities', type=int, default=1,
                        help='Maximum number of instruction entities masked per candidate.')
    parser.add_argument('--mev_weight', type=float, default=0.25,
                        help='Weight of action-contrastive MEV relative to TFV.')
    parser.add_argument('--mev_logprob_temperature', type=float, default=1.0,
                        help='Temperature for mapping MEV log-likelihood margins to [0,1].')

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        'vitbase_r2rfte2e': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5',
        'vitbase_clip': 'pth_vit_base_patch32_224_clip.hdf5',
        'vit-16-ori': 'vit-16.hdf5'
    }
    # args.img_ft_file = ft_file_map[args.features]
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')
    if args.candidate_cap_dir is None:
        args.candidate_cap_dir = os.path.join(ROOTDIR, 'R2R', 'captions')

    if args.dataset == 'rxr':
        args.anno_dir = os.path.join(ROOTDIR, 'RxR', 'annotations')
    else:
        args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # remove unnecessary args
    if args.dataset != 'rxr':
        del args.langs

    return args
