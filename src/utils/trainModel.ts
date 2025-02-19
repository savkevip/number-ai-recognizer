import * as tf from "@tensorflow/tfjs";
import { createModel } from "./tfModel";
import { loadMnistData } from "./mnistData";
import { BATCH_SIZE, EPOCHS } from "./constants";

export async function trainModel(
  onProgress?: (
    epoch: number | null,
    batch: number,
    totalBatches: number
  ) => void
) {
  console.log("🔄 Training started...");

  const model = createModel();
  const data = await loadMnistData();

  const hasNaNImages = tf.any(tf.isNaN(data.trainImages)).arraySync();
  const hasNaNLabels = tf.any(tf.isNaN(data.trainLabels)).arraySync();

  if (hasNaNImages || hasNaNLabels) {
    console.error("⚠️ NaN detected in dataset! Check mnistData.ts!");
    return;
  }

  const totalBatches = Math.ceil(data.trainImages.shape[0] / BATCH_SIZE);

  await model.fit(data.trainImages, data.trainLabels, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationSplit: 0.2,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `✅ Epoch ${epoch + 1}/${EPOCHS} - Loss: ${logs?.loss.toFixed(4)}`
        );
        if (onProgress) onProgress(epoch + 1, 0, totalBatches);
      },
      onBatchEnd: async (batch, logs) => {
        console.log(
          `🔹 Batch ${
            batch + 1
          }/${totalBatches} - Accuracy: ${logs?.acc?.toFixed(2)}`
        );
        if (onProgress) onProgress(null, batch + 1, totalBatches);
      },
    },
  });

  console.log("✅ Training completed!");

  return model;
}
