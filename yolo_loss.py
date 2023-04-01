import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from zmq import has


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
    # So we are comparing all boxes to all boxes ooooh
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    # left top corners given by x1, y1. 
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    # right bottom corner given by x2, y2
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S                # width and height of initial grid
        self.B = B                # number of boxes per grid, determines pred_boxes_list length
        self.l_coord = l_coord    # the value of lambdas
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        input to compute_iou
        boxes: (N,4) representing by x,y,w,h
        These are scaled by S, so that x = S*(x1+w/2)

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # I imagine that x, y are in [0,S) which in this case is 14
        S = self.S
        x, y, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h
        x2, y2 = x/S + 0.5*w, y/S + 0.5*h
        return torch.hstack((x1.reshape(-1,1),y1.reshape(-1,1),x2.reshape(-1,1),y2.reshape(-1,1)))

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters: Changed 4 to 5 as it makes more sense for the predicted boxes to keep their confidence
        for later calculations like noobj_loss while box_target is GT so it must be 1. M == torch.sum(has_object_map)
        pred_box_list : [(tensor) size (M, 5) ...] 2 corners, each 2 coords, confidence
        box_target : (tensor)  size (M, 4)

        Returns: In xywh format
        best_iou: (tensor) size (M, 1)
        best_boxes : (tensor) size (M, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### CODE ###
        # N, S, S information was lost but they are all indexed corresponding to each other so not an issue
        M = box_target.shape[0]
        best_ious = torch.zeros((M,1)).to(device)
        best_boxes = torch.zeros((M,5)).to(device)
        for cell in range(M):
            ground_truth_box = box_target[cell].reshape(-1,4)
            candidate_boxes = torch.zeros((self.B,4)).to(device)
            for box in range(self.B):
                candidate_boxes[box] = pred_box_list[box][cell,:4]
            # This returns a (self.B, 1) array so now we pick the max IOU
            iou = compute_iou(self.xywh2xyxy(candidate_boxes), self.xywh2xyxy(ground_truth_box))
            best_iou, best_idx = torch.max(iou, 0)
            best_ious[cell] = best_iou
            best_boxes[cell] = pred_box_list[best_idx][cell]
        return best_ious, best_boxes

    #5#
    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20 classes)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        # L2 loss and only over gridcells which have object
        loss_matrix = ((classes_pred-classes_target)**2)[has_object_map]
        return torch.sum(loss_matrix)
    
    #4#
    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        # This has B=2 so pred_boxes_list
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S) is 1 if it has object and 0 if not 

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        total_loss = 0
        for i in range(self.B):
            # We go through set of boxes and take the last column: objectability
            pred_boxes = pred_boxes_list[i][:,:,:,-1] #(N, S, S)
            # GT = 0 so we do squared of what we predicted and doesn't have object
            total_loss += torch.sum(pred_boxes[~has_object_map]**2)
        return self.l_noobj * total_loss

    #3#
    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient
        If this is ground truth then confidence is 1, why even bother pass it as argument?
        """
        ### CODE
        return torch.sum((box_pred_conf-1)**2)

    #1,2#
    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters: xywh
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        lets_differ_wholly = (box_pred_response[:,:2] - box_target_response[:,:2])**2
        lets_differ_sqarly = (torch.sqrt(box_pred_response[:,2:]) - torch.sqrt(box_target_response[:,2:]))**2
        return self.l_coord * torch.sum(lets_differ_wholly + lets_differ_sqarly)

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0) # batch

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction) last 20 in 4th dimension size (N, S, S, 20)
        pred_boxes_list = []
        for i in range(self.B):
            pred_boxes_list.append(pred_tensor[:,:,:,5*i:5*(i+1)])
        pred_cls = pred_tensor[:,:,:,-20:]

        #5# compute classification loss
        class_prediction_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)/N

        #4# compute no-object loss
        no_object_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)/N

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # Into occ_pred_boxes (-1, 5) and occ_target_boxes (-1, 4)
        # 1) only keep having-object cells (this will reduce SxS as most grid cells dont have object)
        # 2) vectorize all dimensions except for the last one (5)

        occ_pred_boxes = []
        for i in range(self.B):
            occ_pred_boxes.append(pred_boxes_list[i][has_object_map])
        occ_target_boxes = target_boxes[has_object_map]
        
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(occ_pred_boxes, occ_target_boxes)

        #1,2# compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        regression_loss = self.get_regression_loss(best_boxes[:,:4], occ_target_boxes)/N

        #3# compute contain_object_loss
        contain_conf_loss = self.get_contain_conf_loss(best_boxes[:,-1], 0)/N

        # compute final loss
        final_loss = regression_loss + contain_conf_loss + no_object_loss + class_prediction_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=final_loss,
            reg_loss=regression_loss,
            containing_obj_loss=contain_conf_loss,
            no_obj_loss=no_object_loss,
            cls_loss=class_prediction_loss,
        )
        return loss_dict
