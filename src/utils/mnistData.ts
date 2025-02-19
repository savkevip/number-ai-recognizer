import * as tf from "@tensorflow/tfjs";
import {
  DEFAULT_BATCH_SIZE,
  IMAGE_SIZE,
  IMAGE_SIZES,
  MNIST_IMAGES_URL,
  MNIST_LABELS_URL,
  NUM_CLASSES,
  NUM_DATASET_ELEMENTS,
  NUM_TEST_ELEMENTS,
  NUM_TRAIN_ELEMENTS,
} from "./constants";

async function loadMnistImages(
  batchSize: number = DEFAULT_BATCH_SIZE
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = MNIST_IMAGES_URL;
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const dataset = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZES);

      const canvas = document.createElement("canvas");
      canvas.width = IMAGE_SIZE;
      canvas.height = IMAGE_SIZE;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject("No canvas!");
        return;
      }

      console.log(
        `✅ Starting with loading MNIST dataset u batchs -> batch size: ${batchSize}`
      );

      let processedImages = 0;

      function processBatch(startIndex: number) {
        console.log(
          `📦 Batch processing ${startIndex} - ${startIndex + batchSize}`
        );

        for (
          let i = startIndex;
          i < Math.min(startIndex + batchSize, NUM_DATASET_ELEMENTS);
          i++
        ) {
          const row = Math.floor(i / 1000);
          const col = i % 1000;

          ctx?.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
          ctx?.drawImage(
            img,
            col * 28,
            row * IMAGE_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            0,
            0,
            IMAGE_SIZE,
            IMAGE_SIZE
          );

          const imageData = ctx?.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
          const pixels = imageData?.data ?? [];

          for (let j = 0; j < IMAGE_SIZES; j++) {
            dataset[i * IMAGE_SIZES + j] = pixels[j * 4] / 255;
          }
        }

        processedImages += batchSize;
        if (processedImages < NUM_DATASET_ELEMENTS) {
          requestAnimationFrame(() => processBatch(processedImages));
        } else {
          console.log("✅ Loading MNIST dataset done!");
          resolve(dataset);
        }
      }

      processBatch(0);
    };
    img.onerror = (err) => reject(err);
  });
}

async function loadMnistLabels(): Promise<Uint8Array> {
  const response = await fetch(MNIST_LABELS_URL);
  const labelBuffer = await response.arrayBuffer();
  return new Uint8Array(labelBuffer);
}

export async function loadMnistData() {
  console.log("📥 Loading MNIST dataset...");

  const images = await loadMnistImages();
  const labels = await loadMnistLabels();

  const imagesTensor = tf.tensor2d(images, [NUM_DATASET_ELEMENTS, IMAGE_SIZES]);
  const labelsTensor = tf.oneHot(tf.tensor1d(labels, "int32"), NUM_CLASSES);

  return {
    trainImages: imagesTensor.slice([0, 0], [NUM_TRAIN_ELEMENTS, IMAGE_SIZES]),
    trainLabels: labelsTensor.slice([0, 0], [NUM_TRAIN_ELEMENTS, NUM_CLASSES]),
    testImages: imagesTensor.slice(
      [NUM_TRAIN_ELEMENTS, 0],
      [NUM_TEST_ELEMENTS, IMAGE_SIZES]
    ),
    testLabels: labelsTensor.slice(
      [NUM_TRAIN_ELEMENTS, 0],
      [NUM_TEST_ELEMENTS, NUM_CLASSES]
    ),
  };
}
