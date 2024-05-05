import os

from data_processing import drawing_face_box, select_negative_samples
from feature_extraction import resolve_haar_feature
from training import train_model
from face_detection import pyramid, sliding_window, compute_iou, evaluate_image
from evaluation import plot_roc_curve


image_folder = 'D:\Code\Homework\data\Caltech_WebFaces'
labels_file = 'D:\Code\Homework\data\WebFaces_GroundThruth.txt'

def main(folder, file):
    face_box =  drawing_face_box(file)

    select_negative_samples(folder, face_box, num_samples=2)
    negative_samples = select_negative_samples(folder, face_box, num_samples=2)
    features = []
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        features.append(tuple(resolve_haar_feature(image_path, feature_type, face_box[image_name]) for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']))

    negative_features = []
    for image_name in os.listdir(folder):  # Iterate over each image in the folder
        image_path = os.path.join(folder, image_name)  # Get the full path of the image
        bounding_box_1, bounding_box_2 = negative_samples[image_name]
        negative_features.append(tuple(resolve_haar_feature(image_path, feature_type, bounding_box_1) for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']))
        negative_features.append(tuple(resolve_haar_feature(image_path, feature_type, bounding_box_2) for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']))

    train_model(features, negative_features)

    evaluate_image(image, model, scale, stepSize, windowSize, face_boxes, iou_threshold)

    plot_roc_curve(y_true, y_pred_proba)


if __name__ == "__main__":
    main(image_folder, labels_file)
