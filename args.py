
import argparse
import os
import numpy as np
import time
import datetime



data_dir = os.path.abspath(os.path.dirname(__file__)) + '/_data/'

def get_args(description='ProST on Retrieval Task'):

    def time2file_name(time):
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        hour = time[11:13]
        minute = time[14:16]
        second = time[17:19]
        time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
        return time_filename

    rand_wait = np.random.randint(low=1, high=20)
    time.sleep(rand_wait)
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", default=0, type=int, help="Whether to run training.")
    parser.add_argument("--do_train", default=1, type=int, help="Whether to run training.")
    parser.add_argument("--do_eval", default=0, type=int, help="Whether to run eval on the dev set.")
    parser.add_argument("--from_script", default=0, type=int)

    parser.add_argument('--train_csv', type=str, default='')
    parser.add_argument('--val_csv', type=str, default='')
    parser.add_argument('--data_path', type=str, default='', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=os.path.abspath(os.path.dirname(__file__))+f'/ckpts/{date_time}', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", default=0, type=int, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', default=False, 
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt9k", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--adap_para', type=float, default=0.2, help='coefficient for bert branch.')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', default=0, type=int, help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', default=0, type=int, help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', default=1, type=int, help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', default=0, type=int)

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="seqTransf",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--best_ckpt_path", default="", type=str, help="Choose a ckpt to use")
    parser.add_argument("--eval_in_train", default=1, type=int, help="Whether to do eval in training.")

    parser.add_argument('--pooling_head', type=int, default=1, help='Number of parallel heads for pooling frames.')
    parser.add_argument('--pooling_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling.')

    # Stochastic Text Modeling
    parser.add_argument('--support_loss_weight',  type=float, default=0.8, help='compute the contrastive between pooled-video and support text embedding, default=0.')
    parser.add_argument('--stochastic_prior', type=str, default='uniform01', choices=['uniform01', 'normal'], help="use which prior for the re-parameterization, default to unifrom01")
    parser.add_argument('--stochastic_prior_std',  type=float, default=1.0, help='std value for the reprameterization prior')
    parser.add_argument('--stochasic_trials', type=int, default=20, help='perform [stochastic trials] to compute the averaged text embedding at validation')

    # GNN
    parser.add_argument('--gnn_type', type=str, default='rgat', choices=['rgat', 'gat'])
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--gnn_nheads', type=int, default=4)
    parser.add_argument('--gnn_leaky_slope', type=float, default=0.1)
    parser.add_argument('--gnn_nrels', type=int, default=3)
    parser.add_argument('--gnn_v2', type=int, default=0)
    parser.add_argument('--framepe', type=int, default=1)

    # EBM
    parser.add_argument('--mcmc_coef_reg', default=1., type=float)
    parser.add_argument('--mcmc_steps', default=20, type=int)
    parser.add_argument('--mcmc_step_size', default=1., type=float)
    parser.add_argument('--mcmc_noise', default=0.005, type=float)
    parser.add_argument('--max_buffer_vol', default=500, type=int)
    parser.add_argument('--eam_loss_weight', default=1., type=float)
    parser.add_argument('--eam_support_loss_weight', default=1., type=float)
    parser.add_argument('--energy_fn', type=str, default='mlp', choices=['mlp', 'bilinear', 'cossim'])
    parser.add_argument('--energy_pooling', type=str, default='avg', choices=['avg', 'max', 'min'])

    # SigLoss
    parser.add_argument('--loss_fn', type=str, default='sig', choices=['clip', 'sig'])
    parser.add_argument('--fix_sig_loss_param', type=int, default=0, help='')
    parser.add_argument('--temperature', type=float, default=4.77, help='')
    parser.add_argument('--bias', type=float, default=-12.93, help='')

    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # Set dataset-specific parameters
    if not args.from_script:
        if args.datatype == 'didemo':
            args.data_path = f'{data_dir}/DiDeMo/annotation/'
            args.features_path = f'{data_dir}/DiDeMo/videos/'
            args.max_words = 64
            args.max_frames = 64
            args.gradient_accumulation_steps = 4

        elif args.datatype.startswith('msrvtt'):
            args.data_path = f'{data_dir}/MSRVTT/msrvtt_data/MSRVTT_data.json'
            if args.datatype == 'msrvtt9k':
                args.train_csv = f'{data_dir}/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv'
            elif args.datatype =='msrvtt7k':
                args.train_csv = f'{data_dir}/MSRVTT/msrvtt_data/MSRVTT_train.7k.csv'
            else:
                raise NotImplementedError
            args.expand_msrvtt_sentences = 1
            args.val_csv = f'{data_dir}/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv'
            args.features_path = f'{data_dir}/MSRVTT/videos/all/'
            args.max_words = 32
            args.max_frames = 12

        elif args.datatype.startswith('msvd'):
            args.data_path = f'{data_dir}/MSVD/msvd_data/'
            args.features_path = f'{data_dir}/MSVD/videos/'
            args.max_words = 32
            args.max_frames = 12

        elif args.datatype.startswith('vatex'):
            args.data_path = f'{data_dir}/VATEX/vatex_data/'
            args.features_path = f'{data_dir}/VATEX/videos/'
            args.max_words = 32
            args.max_frames = 12

        else:
            raise NotImplementedError

    clip_arch = args.pretrained_clip_name

    if not args.from_script:
        args.output_dir = os.path.abspath(os.path.dirname(__file__)) + \
            f'/ckpts/{args.datatype}/' + \
            f'{clip_arch.replace("/", "_").replace("-", "_")}-{date_time}/'


    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args
