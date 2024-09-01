import matplotlib.pyplot as plt
import numpy as np

def plot_results_from_output(output):
    cmap = plt.get_cmap('viridis')  # Color map for distinguishing more categories
    for image_result in output:
        image = np.array(image_result['images'])
        masks = np.array(image_result['masks'])
        boxes = np.array(image_result['boxes'])
        labels = image_result['labels']

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if image.shape[0] == 3:  # Assuming the format is CHW
            image = image.transpose(1, 2, 0)
        
        ax.imshow(image)
        
        unique_classes = np.unique(masks)
        colored_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
        
        for cls in unique_classes:
            if cls == 0:
                continue  # Ignore background
            color = cmap(cls / len(unique_classes))[:3]
            colored_mask[masks == cls] = color

        # Expand masks to (256, 256, 1) to match the shape of colored_mask
        masks_expanded = np.repeat(masks[:, :, np.newaxis], 3, axis=2)
        masked_colored_mask = np.ma.masked_where(masks_expanded == 0, colored_mask)

        ax.imshow(masked_colored_mask, alpha=0.3)  # Make it more transparent
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, str(label), color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
        
        plt.show()

import matplotlib.patches as mpatches

def visualize_masks(pred_mask, gt_mask, image=None):
    """
    Visualize the predicted mask and ground truth mask by overlapping them with different colors.
    
    Parameters:
    - pred_mask (numpy array): Predicted mask with shape (H, W), values should represent category IDs.
    - gt_mask (numpy array): Ground truth mask with shape (H, W), values should represent category IDs.
    - image (numpy array, optional): Original image with shape (H, W, 3) to overlay the masks on. Default is None.
    """
    unique_categories = np.unique(np.concatenate((pred_mask, gt_mask)))
    n_categories = len(unique_categories)

    # Create a color map with distinct colors for each category
    cmap = plt.get_cmap('tab20', n_categories)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    if image is not None:
        ax[0].imshow(image)
    ax[0].imshow(gt_mask, cmap=cmap, alpha=0.5)
    ax[0].set_title('Ground Truth')

    if image is not None:
        ax[1].imshow(image)
    ax[1].imshow(pred_mask, cmap=cmap, alpha=0.5)
    ax[1].setTitle('Predicted')

    # Overlay ground truth and predicted masks
    overlay = np.zeros_like(pred_mask, dtype=float)
    overlay[pred_mask == gt_mask] = pred_mask[pred_mask == gt_mask]
    overlay[pred_mask != gt_mask] = pred_mask[pred_mask != gt_mask] + 0.5  # Add offset for differentiation

    if image is not None:
        ax[2].imshow(image)
    ax[2].imshow(overlay, cmap=cmap, alpha=0.5)
    ax[2].setTitle('Overlay')

    # Create a legend
    handles = [mpatches.Patch(color=cmap(i), label=f'Category {cat_id}') for i, cat_id in enumerate(unique_categories)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 0.9))
    
    plt.show()
