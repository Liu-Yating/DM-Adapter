import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)


    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def compute_imkt(text_features, pid):
    imkt_loss = 0
    pid = pid.detach().cpu().numpy()
    for p in np.unique(pid):
        mask = pid == p
        p_matrix = text_features[mask]
        p_norm= p_matrix / p_matrix.norm(dim=-1, keepdim=True)
        imkt_loss = imkt_loss + torch.sum(torch.cdist(p_norm, p_norm, p=2.0))
    return imkt_loss


def compute_triplet(image_features, text_features):
    #image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    #text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    # compute image-sentence score matrix
    # scores = torch.cdist(image_norm, text_norm, p=2.0)
    scores = torch.cdist(image_features, text_features, p=2.0)

    diagonal = scores.diag().view(image_features.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    dis_s = (1 - scores + d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    dis_im = (1 - scores + d2).clamp(min=0)
    # compare every diagonal score to scores in its column

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    dis_s = dis_s.masked_fill_(I, 0)
    dis_im = dis_im.masked_fill_(I, 0)

    if True:
        dis_s = dis_s.max(1)[0]
        dis_im = dis_im.max(0)[0]

    #return dis_s.sum() + dis_im.sum()
    return dis_s.mean() + dis_im.mean()


def compute_triplet_enhance(image_features, text_features, pid):

    outer_loss = compute_triplet(image_features, text_features)
    inner_loss = 0
    pid = pid.detach().cpu().numpy()
    for p in np.unique(pid):
        mask = pid == p
        text_features_mask = text_features[mask]
        image_features_mask = image_features[mask]
        inner_loss = inner_loss + compute_triplet(text_features_mask, image_features_mask)
    return outer_loss + inner_loss /len(pid)

def compute_triplet_enhance_shuffle(image_features, text_features, pid):
    batchsize = len(image_features)
    outer_loss = compute_triplet(image_features, text_features)
    inner_loss = 0
    pid = pid.detach().cpu().numpy()
    for p in np.unique(pid):
        mask = pid[:int(batchsize/2)] == p
        pad = np.zeros(int(batchsize/2), dtype=bool)
        mask = np.concatenate((mask, pad))
        text_features_mask = text_features[mask]
        image_features_mask = image_features[mask]
        inner_loss = inner_loss + compute_triplet(text_features_mask, image_features_mask)
    return outer_loss + inner_loss /len(pid)
