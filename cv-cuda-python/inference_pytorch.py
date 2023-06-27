import os
import json
import argparse
import time

import torch
from PIL import Image
from torchvision import transforms, models


class ClassificationSample:
    def __init__(
        self,
        input_path,
        labels_file,
        batch_size,
        target_img_height,
        target_img_width,
        device_id,
    ):
        self.input_path = input_path
        self.batch_size = batch_size
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.device_id = device_id
        self.labels_file = labels_file

        # tag: Image Loading
        # Start by parsing the input_path expression first.
        if os.path.isfile(self.input_path):
            # Read the input image file.
            self.file_names = [self.input_path]
            # Then create a dummy list with the data from the same file to simulate a
            # batch.
            self.data = [open(path, "rb").read() for path in self.file_names]

        elif os.path.isdir(self.input_path):
            # It is a directory. Grab all the images from it.
            self.file_names = glob.glob(os.path.join(self.input_path, "*.jpg"))
            self.data = [open(path, "rb").read() for path in self.file_names]
            print("Read a total of %d JPEG images." % len(self.data))

        else:
            print(
                "Input path not found. "
                "It is neither a valid JPEG file nor a directory: %s" % self.input_path
            )
            exit(1)

        # tag: Validate other inputs
        if not os.path.isfile(self.labels_file):
            print("Labels file not found: %s" % self.labels_file)
            exit(1)

        if self.batch_size <= 0:
            print("batch_size must be a value >=1.")
            exit(1)

        if self.target_img_height < 10:
            print("target_img_height must be a value >=10.")
            exit(1)

        if self.target_img_width < 10:
            print("target_img_width must be a value >=10.")
            exit(1)

    def run(self):
        data_transform = transforms.Compose(
            [transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        file_name_batches = [
            self.file_names[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.file_names), self.batch_size)
        ]
        data_batches = [
            self.data[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.data), self.batch_size)
        ]

        if self.batch_size == 1:
            effective_batch_size = 1
            img = Image.open(self.file_names[0])

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0).cuda()


        # create model
        model = models.resnet50(pretrained=True)
        model.to("cuda")

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            infer_output = model(img).cpu()
        
        # tag: Postprocess
        """
        Postprocessing function normalizes the classification score from the network
        and sorts the scores to get the TopN classification scores.
        """
        # Apply softmax to Normalize scores between 0-1
        scores = torch.nn.functional.softmax(infer_output, dim=1)

        # Sort output scores in descending order
        _, indices = torch.sort(infer_output, descending=True)

        # tag: Display Top N Results
        # Read and parse the classes
        with open(self.labels_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # top results to print out
        topN = 5
        for img_idx in range(effective_batch_size):
            print(
                "Result for the image: %d of %d"
                % (img_idx + 1, effective_batch_size)
            )

            # Display Top N Results
            for idx in indices[img_idx][:topN]:
                idx = idx.item()
                print(
                    "\tClass : ",
                    classes[idx],
                    " Score : ",
                    scores[img_idx][idx].item(),
                )


def main():
    parser = argparse.ArgumentParser(
        "Classification sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        default="cat.jpg",
        type=str,
        help="Either a path to a JPEG image or a directory containing JPEG "
        "images to use as input.",
    )

    parser.add_argument(
        "-l",
        "--labels_file",
        default="imagenet-classes.txt",
        type=str,
        help="The labels file to read and parse.",
    )

    parser.add_argument(
        "-th",
        "--target_img_height",
        default=224,
        type=int,
        help="The height to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-tw",
        "--target_img_width",
        default=224,
        type=int,
        help="The width to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="Input Batch size"
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Run the sample.
    sample = ClassificationSample(
        args.input_path,
        args.labels_file,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
    )
    st = time.time()
    for i in range(10):
        print("The {}th times".format(i))
        sample.run()
    et = time.time()
    print("the average time is ", str((et - st)/10))


if __name__ == '__main__':
    main()
