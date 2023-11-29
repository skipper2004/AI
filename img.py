from fastsam import FastSAM, FastSAMPrompt
import torch
import cv2
import numpy as np


def main(name_model: str = "FastSAM-x.pt", name_image: str = "dogs.jpg", prompt: str = "a dog") -> None:
    model = FastSAM("./weights/"+name_model)

    img = cv2.imread("./images/" + name_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    everything_results = model(
        img,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )

    prompt_process = FastSAMPrompt(img, everything_results, device=device)

    ann = prompt_process.text_prompt(text=prompt)

    prompt_process.plot(
        annotations=ann,
        output_path="./output/"+name_image,
        bboxes=None,
        points=None,
        point_label=None,
        withContours=False,
        better_quality=False,
    )

    # show
    original_image = cv2.imread("./images/" + name_image)
    masked_image = cv2.imread("./output/"+name_image)

    result_image = np.concatenate((original_image, masked_image), axis=1)
    cv2.imshow("Original vs Masked", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
