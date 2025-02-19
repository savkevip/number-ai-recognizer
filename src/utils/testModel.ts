import * as tf from "@tensorflow/tfjs";
import { loadMnistData } from "./mnistData";
import { IMAGE_SIZE } from "./constants";

export async function testModel(model: tf.Sequential) {
  console.log("🧪 Model testing...");

  const data = await loadMnistData();
  const testSample = data.testImages.slice(
    [0, 0],
    [1, IMAGE_SIZE * IMAGE_SIZE]
  );

  const prediction = model.predict(testSample) as tf.Tensor;
  const predictionArray = (await prediction.array()) as number[][];

  console.log("Prediction:", predictionArray[0]);
  console.log(
    "Model thinks the number is:",
    predictionArray[0].indexOf(Math.max(...predictionArray[0]))
  );
}
