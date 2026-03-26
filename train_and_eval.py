from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time

import time


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, logger, local_rank=0):
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        # (bs, nclips, M), (bs, nclips, M), (bs, nclips, M), (bs, nclips, F, T, C, H, W), (bs, nclips, F)
        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss.item())
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss.item()),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, v_feat_list):
    sim_matrix = []
    for idx1, b1 in enumerate(t_mask_list):

        input_mask, segment_ids, *_tmp = b1
        sequence_output = t_feat_list[idx1]
        each_row = []

        for idx2, b2 in enumerate(v_mask_list):
            video_mask, *_tmp = b2
            visual_output = v_feat_list[idx2]
            b1b2_logits_1, _ = model.get_max_similarity_logits(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits_1
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)

        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
        print("{}/{}\r".format(idx1, len(t_mask_list)), end="")

    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu, logger):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        start_1 = time.time()
        for bid, batch in enumerate(test_dataloader):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask)

                batch_sequence_output_list.append(text_feat)
                batch_list_t.append((input_mask, segment_ids,))
                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    video_feat = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(video_feat)

                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                text_feat, video_feat = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(text_feat)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(video_feat)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        end_1 = time.time()
        cache_feature_time = end_1 - start_1
        logger.info("cache_feature_time: {}".format(cache_feature_time))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        start_2 = time.time()
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        end_2 = time.time()
        compute_sim_time = end_2 - start_2
        logger.info("compute_sim_time: {}".format(compute_sim_time))

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("------------------------------------------------------------")
    if multi_sentence_:
        logger.info(" All T2V: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info(" All V2T: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
        final_output_str_t2v = f"{tv_metrics['R1']:.1f},{tv_metrics['R5']:.1f},{tv_metrics['R10']:.1f},{tv_metrics['MR']:.1f},{tv_metrics['MeanR']:.1f}"
        final_output_str_v2t = f"{vt_metrics['R1']:.1f},{vt_metrics['R5']:.1f},{vt_metrics['R10']:.1f},{vt_metrics['MR']:.1f},{vt_metrics['MeanR']:.1f}"
    else:
        logger.info(" All T2V: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info(" All V2T: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
        final_output_str_t2v = f"{tv_metrics['R1']:.1f},{tv_metrics['R5']:.1f},{tv_metrics['R10']:.1f},{tv_metrics['MR']:.1f},{tv_metrics['MeanR']:.1f}"
        final_output_str_v2t = f"{vt_metrics['R1']:.1f},{vt_metrics['R5']:.1f},{vt_metrics['R10']:.1f},{vt_metrics['MR']:.1f},{vt_metrics['MeanR']:.1f}"
    logger.info("------------------------------------------------------------")
    logger.info("T2V:")
    logger.info(f'  {final_output_str_t2v}')
    logger.info("V2T:")
    logger.info(f'  {final_output_str_v2t}')
    logger.info("Overall:")
    logger.info(f'  {final_output_str_t2v}, {final_output_str_v2t}')
    logger.info("------------------------------------------------------------")
    R1 = tv_metrics['R1']
    return R1, sim_matrix, [cache_feature_time, compute_sim_time]
