import * as tf from "@tensorflow/tfjs";
import { getCanvasData } from "./extractCanvasData";
import { IMAGE_SIZES, MODEL_INPUT } from "./constants";

export async function recognizeFromCanvas(
  model: tf.Sequential,
  canvas: HTMLCanvasElement
) {
  const pixels = getCanvasData(canvas);

  if (pixels.length !== IMAGE_SIZES) {
    console.error(
      `Error: length of the pixels: ${pixels.length}, expected ${MODEL_INPUT}`
    );
    return;
  }

  const inputTensor = tf.tensor2d([pixels]).reshape([1, MODEL_INPUT]);

  const prediction = model.predict(inputTensor);

  if (!prediction) {
    console.error("❌ Model prediction failed!");
    return;
  }

  const predictionTensor = Array.isArray(prediction)
    ? prediction[0]
    : prediction;

  try {
    const rawPrediction = await predictionTensor.array();
    const predictionArray = rawPrediction as number[][];

    if (!predictionArray || predictionArray.length === 0) {
      console.error("❌ Invalid prediction output:", predictionArray);
      return;
    }

    const predictedNumber = predictionArray[0].indexOf(
      Math.max(...predictionArray[0])
    );

    console.log("Prediction from Canvas:", predictionArray[0]);
    console.log("Choosen number:", predictedNumber);
    alert(`Model thinks you draw this number: ${predictedNumber}`);
  } catch (error) {
    console.error("❌ Error in prediction:", error);
  }
}
