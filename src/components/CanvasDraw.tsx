import { recognizeFromCanvas } from "@/utils/recognizeFromCanvas";
import { testModel } from "@/utils/testModel";
import { trainModel } from "@/utils/trainModel";
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { EPOCHS } from "@/utils/constants";

export default function CanvasDraw() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [epochProgress, setEpochProgress] = useState(0);
  const [batchProgress, setBatchProgress] = useState(0);
  const [totalBatches, setTotalBatches] = useState(1);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    (async () => {
      await handleTrain();
    })();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const handleTrain = async () => {
    setTraining(true);

    const trainedModel = await trainModel((epoch, batch, total) => {
      if (epoch !== null) setEpochProgress(epoch);
      if (batch !== null) setBatchProgress(batch);
      if (total) setTotalBatches(total);
    });

    if (trainedModel) {
      setModel(trainedModel);
    }
    setTraining(false);
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.strokeStyle = "black";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setDrawing(false);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.closePath();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const handleTest = async () => {
    if (!model) {
      alert("Model is not trained yet!");
      return;
    }
    await testModel(model);
  };

  const handlePredict = () => {
    if (!model || !canvasRef.current) {
      alert("Model is not trained or the canvas is unavailable!");
      return;
    }
    recognizeFromCanvas(model, canvasRef.current);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-semibold text-gray-900 mb-4">
        Draw a Number
      </h1>
      {training ? (
        <div className="w-64 mt-4">
          <h2 className="text-2xl font-semibold text-center text-gray-500 mb-4">
            Model training in progress...
          </h2>
          <p className="text-gray-700">
            Epoch {epochProgress}/{EPOCHS}
          </p>
          <div className="w-full bg-gray-300 rounded-full h-3">
            <div
              className="bg-green-500 h-3 rounded-full transition-all"
              style={{ width: `${(epochProgress / EPOCHS) * 100}%` }}
            ></div>
          </div>

          <p className="text-gray-700 mt-2">
            Batch {batchProgress}/{totalBatches}
          </p>
          <div className="w-full bg-gray-300 rounded-full h-3">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all"
              style={{ width: `${(batchProgress / totalBatches) * 100}%` }}
            ></div>
          </div>
        </div>
      ) : (
        <div className="bg-white shadow-lg rounded-2xl p-6">
          <div className="w-full flex justify-center">
            <canvas
              ref={canvasRef}
              width={200}
              height={200}
              className="border-4 border-gray-700 rounded-lg cursor-crosshair bg-white"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
          </div>
          <div className="flex space-x-4 mt-4">
            <button
              onClick={clearCanvas}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
            >
              Clear
            </button>
            <button
              onClick={handlePredict}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
            >
              Predict
            </button>
            <button
              onClick={handleTest}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
            >
              Test Model
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
