from fastsam import FastSAM, FastSAMPrompt
import torch
import cv2
import numpy as np


def main(name_model: str = "FastSAM-x.pt", video_source: int = 0, prompt: str = "a dog") -> None:
    model = FastSAM("./weights/"+name_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        if len(ann) == 0:
            print("No object detected.")
            continue

        prompt_process.plot(
            annotations=ann,
            output_path="./output/frame_output.jpg",
            bboxes=None,
            points=None,
            point_label=None,
            withContours=False,
            better_quality=False,
        )

        # Get the masked image directly from the prompt_process
        masked_image = cv2.imread("./output/frame_output.jpg")
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        # show
        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result_image = np.concatenate((original_image, masked_image), axis=1)
        cv2.imshow("Original vs Masked", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
