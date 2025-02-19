import * as tf from "@tensorflow/tfjs";
import { MODEL_INPUT, NUMBER_OF_NEURONS } from "./constants";

export function createModel(): tf.Sequential {
  const model = tf.sequential();

  // Ulazni sloj - sada očekuje 784 piksela (28x28 slike)
  model.add(
    tf.layers.dense({
      inputShape: [MODEL_INPUT],
      units: NUMBER_OF_NEURONS, // Veći broj neurona za bolje prepoznavanje
      activation: "relu",
    })
  );

  // Hidden layer
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
