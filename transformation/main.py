from plantcv import plantcv as pcv
import cv2
import matplotlib.pyplot as plt
import argparse
def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()
    return args


def blur(img):
    blurred = pcv.gaussian_blur(
        img=img,
        ksize=(5, 5),
        sigma_x=0
    )
    return blurred

def get_object_region_and_analysis(image, masked):
    analysis_dict ={}
    contours, hierarchy = cv2.findContours(
        masked,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    leaf_contour = max(
    contours,
    key=cv2.contourArea
)
    x, y, w, h = cv2.boundingRect(leaf_contour)
    area = cv2.contourArea(leaf_contour)
    priemeter = cv2.arcLength(leaf_contour,True)
    roi = image[y:y+h, x:x+w]
    analysis_dict["roi_image"]= roi
    analysis_dict["area"]= area
    analysis_dict["priemeter"]= priemeter

    return analysis_dict

def get_mask(img):
    hsv = pcv.rgb2gray_hsv(img,channel="s")
    print(hsv)
    mask = pcv.threshold.binary(
    gray_img=hsv,
    threshold=50,
    object_type="light")
    return mask



def main():
    try:
        args = parse_args()
        image = cv2.imread(args.image_path) 
        blurred = blur(image)
        masked= get_mask(image)
        object_analysis = get_object_region_and_analysis(image, masked)
        roi = object_analysis["roi_image"]
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(blurred)
        axes[1].set_title("Gaussian Blur")
        axes[1].axis("off")

        axes[2].imshow(masked)
        axes[2].set_title("masked")
        axes[2].axis("off")

        axes[3].imshow(roi)
        axes[3].set_title("ROI")
        axes[3].axis("off")

        plt.tight_layout()

        # Save the comparison image
        plt.savefig("gaussian_blur_comparison.png", dpi=300)

        # Show the figure
        plt.show()
    except Exception as e :
        print ("there is an error ", e)

if __name__=="__main__":
    main()