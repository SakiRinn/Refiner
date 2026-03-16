import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils import compute_accuracy, mkdirs
from algorithms.symmetricCE import SCELoss


def normalize_losses(losses_array):
    """Min-max normalization preprocessing for losses"""
    min_loss = np.min(losses_array)
    max_loss = np.max(losses_array)
    normalized_losses = (losses_array - min_loss) / (max_loss - min_loss + 1e-8)
    return normalized_losses


class DualViewGMM:
    """Dual-view GMM for sample reliability assessment.

    This class implements the dual-view GMM separation as described in FedRefiner paper:
    - View 1: Loss-based GMM (separates samples by per-sample CE loss)
    - View 2: Representation-distance-based GMM (separates samples by distance to class centroids)

    The final reliable samples are determined by the intersection of both views.
    """

    def __init__(self, n_components=2, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm_loss = None
        self.gmm_dist = None
        self.class_centroids = {}  # Store class centroids
        self.clean_label_loss = 0  # The label for clean samples in loss-GMM
        self.clean_label_dist = 0  # The label for "close" samples in dist-GMM

    def fit(self, losses, features, targets):
        # ==================== View 1: Loss-based GMM ====================
        losses_normalized = normalize_losses(losses)

        self.gmm_loss = GaussianMixture(n_components=self.n_components, random_state=self.random_state, n_init=3)
        self.gmm_loss.fit(losses_normalized)

        labels_loss = self.gmm_loss.predict(losses_normalized)
        means_loss = self.gmm_loss.means_.flatten()
        self.clean_label_loss = np.argmin(means_loss)  # Lower loss = cleaner

        # ==================== View 2: Distance-based GMM ====================
        # Step 1: Compute class centroids using clean samples from loss-GMM
        clean_mask_loss = labels_loss == self.clean_label_loss
        features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features

        unique_classes = np.unique(targets)

        for c in unique_classes:
            class_mask = (targets == c) & clean_mask_loss
            if np.sum(class_mask) > 0:
                self.class_centroids[c] = np.mean(features_np[class_mask], axis=0)

        # Step 2: Compute distance to nearest class centroid for each sample
        if len(self.class_centroids) > 0:
            distances = self._compute_distances_to_centroids(features_np, targets)

            # Step 3: Fit GMM on distances
            self.gmm_dist = GaussianMixture(n_components=self.n_components, random_state=self.random_state, n_init=3)
            self.gmm_dist.fit(distances)

            labels_dist = self.gmm_dist.predict(distances)
            means_dist = self.gmm_dist.means_.flatten()
            self.clean_label_dist = np.argmin(means_dist)  # Lower distance = closer to centroid = more reliable
        else:
            self.gmm_dist = None
            distances = None

        return self

    def _compute_distances_to_centroids(self, features, targets):
        distances = []

        for i, (feat, target) in enumerate(zip(features, targets)):
            if target in self.class_centroids:
                centroid = self.class_centroids[target]
                # Cosine distance = 1 - cosine_similarity
                feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                cos_sim = np.dot(feat_norm, centroid_norm)
                cos_dist = 1 - cos_sim
                distances.append(cos_dist)
            else:
                # If class centroid not available, assign max distance
                distances.append(1.0)

        return np.array(distances).reshape(-1, 1)

    def predict(self, losses, features, targets):
        # View 1: Loss-based prediction
        losses_normalized = normalize_losses(losses)
        labels_loss = self.gmm_loss.predict(losses_normalized)
        clean_mask_loss = labels_loss == self.clean_label_loss

        # View 2: Distance-based prediction
        if self.gmm_dist is not None and len(self.class_centroids) > 0:
            features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features
            distances = self._compute_distances_to_centroids(features_np, targets)

            labels_dist = self.gmm_dist.predict(distances)
            close_mask_dist = labels_dist == self.clean_label_dist
        else:
            # Fallback: all samples are considered close
            close_mask_dist = np.ones(len(losses), dtype=bool)

        # Intersection of both views
        reliable_mask = clean_mask_loss & close_mask_dist

        return reliable_mask, clean_mask_loss, close_mask_dist


def precompute_dual_view_gmm_with_global_model(global_model, train_dataloader, args):
    # Ensure global model is on GPU
    device = next(global_model.parameters()).device
    if device.type == 'cpu':
        global_model = global_model.cuda()
        device = torch.device('cuda')

    global_model.eval()
    all_losses = []
    all_features = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (x, target, idx) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, out = global_model(x)

            # Ensure dimension compatibility
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            ce_losses = F.cross_entropy(out, target, reduction='none')
            all_losses.extend(ce_losses.cpu().numpy())
            all_features.append(feat.cpu())
            all_targets.extend(target.cpu().numpy())

    # Convert to arrays
    losses_array = np.array(all_losses).reshape(-1, 1)
    features_tensor = torch.cat(all_features, dim=0)
    targets_array = np.array(all_targets)

    # Fit dual-view GMM
    dual_view_gmm = DualViewGMM(n_components=2, random_state=0)
    dual_view_gmm.fit(losses_array, features_tensor, targets_array)

    # Compute statistics
    reliable_mask, clean_mask_loss, close_mask_dist = dual_view_gmm.predict(
        losses_array, features_tensor, targets_array
    )

    clean_count = np.sum(clean_mask_loss)
    noisy_count = len(clean_mask_loss) - clean_count
    reliable_count = np.sum(reliable_mask)
    noise_ratio = noisy_count / len(clean_mask_loss)

    statistics = {
        'clean_count': clean_count,
        'noisy_count': noisy_count,
        'reliable_count': reliable_count,
        'noise_ratio': noise_ratio,
    }

    return dual_view_gmm, losses_array, statistics


def precompute_gmm_with_global_model(global_model, train_dataloader, args):
    """Pre-train GMM using global model (legacy single-view version, kept for compatibility)"""
    # Ensure global model is on GPU
    device = next(global_model.parameters()).device
    if device.type == 'cpu':
        global_model = global_model.cuda()
        device = torch.device('cuda')

    global_model.eval()
    all_losses = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (x, target, idx) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            _, _, out = global_model(x)

            # Ensure dimension compatibility
            if out.dim() == 1:
                out = out.unsqueeze(0)

            ce_losses = F.cross_entropy(out, target, reduction='none')
            all_losses.extend(ce_losses.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Convert to numpy array
    losses_array = np.array(all_losses).reshape(-1, 1)

    # Apply normalization preprocessing to losses
    losses_array = normalize_losses(losses_array)

    # Train GMM
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
    gmm.fit(losses_array)

    # Determine clean/noisy labels
    labels_gmm = gmm.predict(losses_array)
    means = gmm.means_.flatten()
    clean_label = np.argmin(means)

    # Count clean/noisy samples
    clean_count = np.sum(labels_gmm == clean_label)
    noisy_count = len(labels_gmm) - clean_count

    # Calculate noise ratio
    noise_ratio = noisy_count / len(labels_gmm)

    return gmm, clean_count, noisy_count, noise_ratio, losses_array


def train_net_fedrefiner(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None, is_warmup=True):
    """FedRefiner local training function with dual-view GMM"""
    net.cuda()

    # Use SGD optimizer only
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    num_classes = net.l3.out_features if hasattr(net, 'l3') else args.num_class

    # Pre-train dual-view GMM (only in non-warmup phase)
    dual_view_gmm = None
    if not is_warmup:
        dual_view_gmm, losses_array, statistics = precompute_dual_view_gmm_with_global_model(global_model, train_dataloader, args)
        if logger:
            logger.info(f"Client {net_id}: Clean samples (loss-view): {statistics['clean_count']}, "
                       f"Reliable samples (dual-view): {statistics['reliable_count']}, "
                       f"Noise ratio: {statistics['noise_ratio']:.3f}")

    for epoch in range(epochs):
        epoch_loss_collector = []

        for batch_idx, (x, target, idx) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            target = target.long()

            _, feat, out = net(x)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            if is_warmup:
                # Stage 1: Warmup with SCE
                sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
                loss = sce_criterion(out, target)
            else:
                # Stage 2: Use dual-view GMM for noise identification
                loss = train_net_fedrefiner_stage2_with_dual_view_gmm(net, x, target, out, feat, dual_view_gmm, args, num_classes)

            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)


    net.to('cpu')
    return None


def train_net_fedrefiner_stage2_with_single_gmm(net, x, target, out, feat, gmm, args, num_classes):
    """FedRefiner stage 2 training logic with single GMM"""
    # Compute cross-entropy loss
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)

    # Apply same normalization preprocessing to current batch losses
    ce_losses = normalize_losses(ce_losses)

    # Use pre-trained GMM for prediction
    labels_gmm = gmm.predict(ce_losses)
    means = gmm.means_.flatten()
    clean_label = np.argmin(means)
    clean_mask = torch.tensor(labels_gmm == clean_label, dtype=torch.bool, device=out.device)
    noisy_mask = ~clean_mask

    # Calculate model prediction confidence
    probs = F.softmax(out, dim=1)
    confidence, predicted_labels = torch.max(probs, dim=1)

    # Label correction: Correct labels for high-confidence noisy samples
    tao = getattr(args, 'tao', 0.8)  # Confidence threshold
    corrected_targets = target.clone()

    # For noisy samples with confidence above threshold, use model prediction as new label
    high_conf_noisy_mask = noisy_mask & (confidence > tao)
    corrected_targets[high_conf_noisy_mask] = predicted_labels[high_conf_noisy_mask]

    # Calculate SCE loss (using corrected labels)
    sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
    sce_loss = sce_criterion(out, corrected_targets)

    # Calculate contrastive loss (based on loss_2 from TriTAN)
    try:
        contrastive_loss = compute_contrastive_loss(feat, corrected_targets, args)
        # Additional check for abnormal loss values
        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss) or contrastive_loss < 0:
            contrastive_loss = torch.tensor(0.0, device=feat.device, requires_grad=True)
    except Exception as e:
        # If any error occurs during contrastive loss computation, set to 0
        contrastive_loss = torch.tensor(0.0, device=feat.device, requires_grad=True)

    # Total loss
    contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    total_loss = sce_loss + contrastive_weight * contrastive_loss

    return total_loss


def train_net_fedrefiner_stage2_with_dual_view_gmm(net, x, target, out, feat, dual_view_gmm, args, num_classes):
    # Compute cross-entropy loss for current batch
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)
    targets_np = target.detach().cpu().numpy()

    # Use dual-view GMM for prediction
    reliable_mask, clean_mask_loss, close_mask_dist = dual_view_gmm.predict(
        ce_losses, feat.detach(), targets_np
    )

    # Convert to torch tensors
    reliable_mask_torch = torch.tensor(reliable_mask, dtype=torch.bool, device=out.device)
    clean_mask_loss_torch = torch.tensor(clean_mask_loss, dtype=torch.bool, device=out.device)
    close_mask_dist_torch = torch.tensor(close_mask_dist, dtype=torch.bool, device=out.device)

    # Noisy samples: not in reliable set
    noisy_mask = ~reliable_mask_torch

    # Calculate model prediction confidence
    probs = F.softmax(out, dim=1)
    confidence, predicted_labels = torch.max(probs, dim=1)

    # Label correction: Correct labels for high-confidence noisy samples
    tao = getattr(args, 'tao', 0.85)  # Confidence threshold
    corrected_targets = target.clone()

    # For noisy samples with confidence above threshold, use model prediction as new label
    high_conf_noisy_mask = noisy_mask & (confidence > tao)
    corrected_targets[high_conf_noisy_mask] = predicted_labels[high_conf_noisy_mask]

    # Calculate SCE loss (using corrected labels)
    sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
    sce_loss = sce_criterion(out, corrected_targets)

    # Calculate contrastive loss
    try:
        contrastive_loss = compute_contrastive_loss(feat, corrected_targets, args)
        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss) or contrastive_loss < 0:
            contrastive_loss = torch.tensor(0.0, device=feat.device, requires_grad=True)
    except Exception as e:
        contrastive_loss = torch.tensor(0.0, device=feat.device, requires_grad=True)

    # Total loss
    contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    total_loss = sce_loss + contrastive_weight * contrastive_loss

    return total_loss


def train_net_fedrefiner_stage2(net, x, target, out, feat, args, num_classes):
    """FedRefiner stage 2 training logic (original version, kept for comparison)"""
    # Compute cross-entropy loss for GMM
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)

    # If batch size is too small, use SCE directly
    if len(ce_losses) < 2:
        sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
        return sce_criterion(out, target)

    # Use GMM to identify noisy samples
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(ce_losses)
    labels_gmm = gmm.predict(ce_losses)
    means = gmm.means_.flatten()
    clean_label = np.argmin(means)
    clean_mask = torch.tensor(labels_gmm == clean_label, dtype=torch.bool, device=out.device)
    noisy_mask = ~clean_mask

    # Calculate model prediction confidence
    probs = F.softmax(out, dim=1)
    confidence, predicted_labels = torch.max(probs, dim=1)

    # Label correction: Correct labels for high-confidence noisy samples
    tao = getattr(args, 'tao', 0.8)  # Confidence threshold
    corrected_targets = target.clone()

    # For noisy samples with confidence above threshold, use model prediction as new label
    high_conf_noisy_mask = noisy_mask & (confidence > tao)
    corrected_targets[high_conf_noisy_mask] = predicted_labels[high_conf_noisy_mask]

    # Calculate SCE loss (using corrected labels)
    sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
    sce_loss = sce_criterion(out, corrected_targets)

    # Calculate contrastive loss (based on loss_2 from TriTAN)
    contrastive_loss = compute_contrastive_loss(feat, corrected_targets, args)

    # Total loss
    contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    total_loss = sce_loss + contrastive_weight * contrastive_loss

    return total_loss


def compute_contrastive_loss(feats, targets, args):
    """Calculate contrastive loss, based on loss_2 from TriTAN - GPU optimized version"""
    # Ensure correct feature dimensions
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    batch_size = feats.shape[0]

    # If batch size is too small, return 0 directly
    if batch_size < 2:
        return torch.tensor(0.0, device=feats.device, requires_grad=True)

    # Sampling strategy: If batch is too large, sample to improve computational efficiency
    max_samples = getattr(args, 'contrastive_max_samples', 64)
    if batch_size > max_samples:
        # Random sampling, maintaining label distribution
        indices = torch.randperm(batch_size, device=feats.device)[:max_samples]
        feats = feats[indices]
        targets = targets[indices]
        batch_size = max_samples

    # Normalize features
    feats = F.normalize(feats, p=2, dim=1)

    # Calculate feature similarity matrix
    sim_mat = torch.matmul(feats, feats.t())

    # Construct label matrix - using vectorized operations
    targets_expanded = targets.unsqueeze(1).expand(-1, batch_size)
    targets_expanded_t = targets_expanded.t()

    # Construct positive/negative sample mask - vectorized operations
    pos_mask = (targets_expanded == targets_expanded_t)
    neg_mask = ~pos_mask

    # Hard negative sample mining - vectorized operations
    hard_neg_mask = neg_mask & (sim_mat > 0.5)
    pos_mask = pos_mask & (sim_mat < (1 - 1e-5))

    # Use torch.where for batch indexing, avoiding direct indexing
    pos_sim = torch.where(pos_mask, sim_mat, torch.zeros_like(sim_mat))
    neg_sim = torch.where(hard_neg_mask, sim_mat, torch.zeros_like(sim_mat))

    # Calculate loss - vectorized summation
    pos_loss = torch.sum(pos_sim)
    neg_loss = torch.sum(neg_sim)

    # Check for valid positive/negative sample pairs
    if pos_loss > 0 and neg_loss > 0:
        contrastive_loss = (pos_loss + neg_loss) / batch_size
    else:
        contrastive_loss = torch.tensor(0.0, device=feats.device, requires_grad=True)

    return contrastive_loss


def fedrefiner_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    """FedRefiner algorithm main function"""
    best_test_acc = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_loss = float('inf')
    record_test_acc_list = []
    record_f1_list = []
    record_precision_list = []
    record_recall_list = []
    record_loss_list = []

    # Get warmup rounds
    warmup_rounds = getattr(args, 'fedrefiner_warmup', 5)

    # Stage 1: Warmup phase (using SCE)
    logger.info("=== FedRefiner Stage 1: Warmup with SCE ===")
    print("=== FedRefiner Stage 1: Warmup with SCE ===")
    for round in range(warmup_rounds):
        logger.info(f"FedRefiner Warmup Round {round}")
        print(f"FedRefiner Warmup Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # Local training (warmup phase)
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            train_net_fedrefiner(net_id, net, global_model, train_dl_local, test_dl, args.epochs,
                               args.lr, args.optimizer, args, device, logger, is_warmup=True)

        # Model aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_w)
        global_model.cuda()

        # Testing
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')

        # Update best results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss

        logger.info(f'Warmup Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Warmup Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')

    logger.info("=== FedRefiner Stage 2: Dual-View GMM + Label Correction + Contrastive Learning ===")
    print("=== FedRefiner Stage 2: Dual-View GMM + Label Correction + Contrastive Learning ===")

    # Stage 2: GMM + Label Correction + Contrastive Learning
    for round in range(warmup_rounds, n_comm_rounds):
        logger.info(f"FedRefiner Stage 2 Round {round}")
        print(f"FedRefiner Stage 2 Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # Local training (stage 2)
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            # Ensure global_model is on GPU for GMM pre-training
            global_model.cuda()
            # Use optimized training function (pre-trained GMM)
            train_net_fedrefiner(net_id, net, global_model, train_dl_local, test_dl, args.epochs,
                               args.lr, args.optimizer, args, device, logger, is_warmup=False)

        # Model aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        if args.disco:  # Discrepancy-aware collaboration
            from disco import get_distribution_difference, disco_weight_adjusting
            distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference, hypo_distribution=global_dist)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)

        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_w)
        global_model.cuda()

        # Testing
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')

        # Update best results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss

        logger.info(f'Stage 2 Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Stage 2 Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')

        # Save model
        if args.save_model:
            import os
            mkdirs(args.modeldir + 'fedrefiner/')
            torch.save(global_model.state_dict(), args.modeldir + 'fedrefiner/' + 'globalmodel' + args.log_file_name + '.pth')

    # Calculate average of last 10 rounds
    def last_k_avg(lst, k=10):
        return np.mean(lst[-k:]) if len(lst) >= k else np.mean(lst)

    avg_acc = last_k_avg(record_test_acc_list)
    avg_f1 = last_k_avg(record_f1_list)
    avg_precision = last_k_avg(record_precision_list)
    avg_recall = last_k_avg(record_recall_list)
    avg_loss = last_k_avg(record_loss_list)

    logger.info(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    print(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')

    return record_test_acc_list, best_test_acc
